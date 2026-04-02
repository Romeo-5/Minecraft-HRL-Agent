/**
 * Bridge Module - WebSocket server for Python communication
 * 
 * This module exposes the Minecraft environment to the Python RL backend.
 * Communication protocol follows a request-response pattern:
 * 
 * Python -> JS: { type: 'step', action: skillId }
 * JS -> Python: { type: 'observation', state: {...}, reward: float, done: bool }
 * 
 * Python -> JS: { type: 'reset' }
 * JS -> Python: { type: 'observation', state: {...} }
 * 
 * Python -> JS: { type: 'get_action_space' }
 * JS -> Python: { type: 'action_space', skills: [...] }
 */

const WebSocket = require('ws');

// RCON config — must match server.properties
const RCON_PORT     = 25575;
const RCON_PASSWORD = 'hrltraining';

// How long (ms) a single skill is allowed to run before being force-aborted
const SKILL_TIMEOUT_MS = 90_000;

// Stuck-detection thresholds
const STUCK_CHECK_INTERVAL_MS = 5_000;   // poll every 5 s
const STUCK_MOVE_THRESHOLD    = 0.5;     // blocks — less than this = "not moved"
const STUCK_L1_MS = 15_000;   // 15 s → jump + walk
const STUCK_L2_MS = 30_000;   // 30 s → mine surrounding blocks
const STUCK_L3_MS = 60_000;   // 60 s → place block beneath + jump
const STUCK_L4_MS = 90_000;   // 90 s → RCON /kill (respawn resets position)

class Bridge {
    constructor(bot, skillManager, port = 8765) {
        this.bot = bot;
        this.skillManager = skillManager;
        this.port = port;
        this.wss = null;
        this.client = null;
        this.episodeSteps = 0;
        this.maxEpisodeSteps = 1000;

        // Repeat-action tracking
        this._lastSkillId    = null;
        this._repeatCount    = 0;
        this.REPEAT_GRACE    = 2;    // allow up to this many consecutive repeats for free
        this.REPEAT_PENALTY  = -0.2; // reward deduction per repeat beyond the grace window

        // Stuck-detection state
        this._lastPos        = null;
        this._lastMoveTime   = Date.now();
        this._stuckLevel     = 0;       // highest escape level already tried
        this._stuckInterval  = null;
        this._skillRunning   = false;   // true while _handleStep is executing a skill
    }

    /**
     * Start the WebSocket server
     */
    start() {
        this.wss = new WebSocket.Server({ port: this.port });

        console.log(`[Bridge] WebSocket server started on port ${this.port}`);
        this._startStuckDetector();
        
        this.wss.on('connection', (ws) => {
            console.log('[Bridge] Python client connected');
            this.client = ws;
            
            ws.on('message', async (message) => {
                try {
                    const request = JSON.parse(message);
                    const response = await this._handleRequest(request);
                    ws.send(JSON.stringify(response));
                } catch (error) {
                    console.error('[Bridge] Error handling message:', error);
                    ws.send(JSON.stringify({
                        type: 'error',
                        message: error.message
                    }));
                }
            });
            
            ws.on('close', () => {
                console.log('[Bridge] Python client disconnected');
                this.client = null;
            });
            
            ws.on('error', (error) => {
                console.error('[Bridge] WebSocket error:', error);
            });
        });
    }

    /**
     * Handle incoming requests from Python
     */
    async _handleRequest(request) {
        switch (request.type) {
            case 'step':
                return await this._handleStep(request.action);
            
            case 'reset':
                return await this._handleReset();
            
            case 'get_action_space':
                return this._handleGetActionSpace();
            
            case 'get_observation':
                return {
                    type: 'observation',
                    state: this._getState()
                };
            
            case 'ping':
                return { type: 'pong' };
            
            default:
                return {
                    type: 'error',
                    message: `Unknown request type: ${request.type}`
                };
        }
    }

    /**
     * Handle a step action from the RL agent
     */
    async _handleStep(skillId) {
        this.episodeSteps++;

        // Hard cap: if a skill hangs longer than SKILL_TIMEOUT_MS, abort it.
        // This prevents a single pathfinder deadlock from stalling training forever.
        // IMPORTANT: store the timer handle so we can clearTimeout when the skill
        // finishes first — otherwise the timer fires 90 s later as a "ghost" and
        // calls pathfinder.stop() on whatever skill is running at that point.
        this._skillRunning = true;
        const skillPromise = this.skillManager.executeSkill(skillId);
        let timeoutHandle;
        const timeoutPromise = new Promise(resolve => {
            timeoutHandle = setTimeout(() => {
                console.warn(`[Bridge] Skill ${skillId} timed out after ${SKILL_TIMEOUT_MS / 1000}s — aborting`);
                try { this.bot.pathfinder.stop(); } catch (_) {}
                resolve({ success: false, message: `Skill timed out`, reward: -0.5 });
            }, SKILL_TIMEOUT_MS);
        });

        // Execute the skill; cancel the timer if the skill finishes first
        const result = await Promise.race([skillPromise, timeoutPromise]);
        clearTimeout(timeoutHandle);
        this._skillRunning = false;
        
        // Get new state
        const state = this._getState();
        
        // Check termination conditions
        const done = this._checkDone();
        
        // ── Consecutive repeat penalty ──────────────────────────────────────
        // Penalise calling the same skill back-to-back more than REPEAT_GRACE
        // times in a row. This discourages crafting 10 wooden pickaxes or
        // spamming explore when no progress is being made.
        let repeatPenalty = 0;
        if (skillId === this._lastSkillId) {
            this._repeatCount++;
            if (this._repeatCount > this.REPEAT_GRACE) {
                repeatPenalty = this.REPEAT_PENALTY;
            }
        } else {
            this._lastSkillId = skillId;
            this._repeatCount = 1;
        }
        // ────────────────────────────────────────────────────────────────────

        // Build info dict (DEPS-style error feedback)
        const info = {
            skill_executed: skillId,
            skill_name: this.skillManager.skillRegistry.get(skillId)?.name || 'unknown',
            skill_success: result.success,
            skill_message: result.message,
            steps: this.episodeSteps,
            inventory_count: Object.keys(state.inventory).length,
            repeat_count: this._repeatCount,
            repeat_penalty: repeatPenalty
        };

        return {
            type: 'step_result',
            state: state,
            reward: result.reward + repeatPenalty,
            done: done,
            truncated: this.episodeSteps >= this.maxEpisodeSteps,
            info: info
        };
    }

    /**
     * Handle environment reset
     */
    async _handleReset() {
        this.episodeSteps = 0;
        this._lastSkillId = null;
        this._repeatCount = 0;
        this._resetStuckState();

        // Note: True reset would require server commands or respawn
        // For now, we just return the current state
        console.log('[Bridge] Environment reset (soft reset)');
        
        return {
            type: 'reset_result',
            state: this._getState(),
            info: {
                message: 'Soft reset - bot continues from current position'
            }
        };
    }

    /**
     * Return the action space (skill library)
     */
    _handleGetActionSpace() {
        const skills = this.skillManager.getSkillList();
        
        return {
            type: 'action_space',
            n: skills.length,
            skills: skills
        };
    }

    /**
     * Build the high-level state representation
     * 
     * This follows Plan4MC's approach: represent state as structured metadata
     * rather than raw pixels. The state includes:
     * - Inventory (what items the agent has)
     * - Nearby blocks (what resources are available)
     * - Player stats (health, hunger, position)
     * - Available skills (which actions have met preconditions)
     */
    _getState() {
        const position = this.bot.entity.position;
        
        return {
            // Player stats
            health: this.bot.health,
            food: this.bot.food,
            position: {
                x: Math.floor(position.x),
                y: Math.floor(position.y),
                z: Math.floor(position.z)
            },
            
            // Inventory as {item_name: count}
            inventory: this._getInventory(),
            
            // Nearby blocks (for observation space)
            nearby_blocks: this._getNearbyBlocks(),
            
            // Nearby entities
            nearby_entities: this._getNearbyEntities(),
            
            // Which skills are currently available
            available_skills: this._getAvailableSkills(),
            
            // Time of day in game
            time_of_day: this.bot.time.timeOfDay,
            is_day: this.bot.time.timeOfDay < 13000,
            
            // Biome
            biome: this._getCurrentBiome(),

            // Nearby structures (heuristic detection via signature blocks)
            nearby_structures: this._detectNearbyStructures(),

            // Held item
            held_item: this.bot.heldItem?.name || null
        };
    }

    _getInventory() {
        const inventory = {};
        for (const item of this.bot.inventory.items()) {
            inventory[item.name] = (inventory[item.name] || 0) + item.count;
        }
        return inventory;
    }

    _getNearbyBlocks() {
        const blocks = {};
        const radius = 16;
        const pos = this.bot.entity.position;
        
        // Important block types to track
        const importantBlocks = [
            'oak_log', 'birch_log', 'spruce_log', 'jungle_log',
            'stone', 'cobblestone',
            'iron_ore', 'deepslate_iron_ore',
            'coal_ore', 'deepslate_coal_ore',
            'diamond_ore', 'deepslate_diamond_ore',
            'crafting_table', 'furnace', 'chest',
            'water', 'lava',
            'grass_block', 'dirt', 'sand'
        ];
        
        for (const blockType of importantBlocks) {
            const found = this.bot.findBlocks({
                matching: (b) => b.name === blockType,
                maxDistance: radius,
                count: 10
            });
            
            if (found.length > 0) {
                blocks[blockType] = {
                    count: found.length,
                    nearest_distance: Math.min(...found.map(p => 
                        pos.distanceTo(this.bot.blockAt(p).position)
                    ))
                };
            }
        }
        
        return blocks;
    }

    _getNearbyEntities() {
        const entities = {};
        const radius = 32;
        
        for (const entity of Object.values(this.bot.entities)) {
            if (entity === this.bot.entity) continue;
            
            const distance = entity.position.distanceTo(this.bot.entity.position);
            if (distance <= radius) {
                const type = entity.name || entity.username || 'unknown';
                if (!entities[type]) {
                    entities[type] = { count: 0, nearest_distance: distance };
                }
                entities[type].count++;
                entities[type].nearest_distance = Math.min(
                    entities[type].nearest_distance, 
                    distance
                );
            }
        }
        
        return entities;
    }

    _getAvailableSkills() {
        return this.skillManager.getSkillList()
            .filter(s => s.available)
            .map(s => s.id);
    }

    _getCurrentBiome() {
        try {
            const pos = this.bot.entity.position;
            const block = this.bot.blockAt(pos);
            const biomeId = block?.biome?.id;
            if (biomeId !== undefined && biomeId !== null) {
                // Look up name via minecraft-data (mineflayer 4.x doesn't populate biome.name)
                const mcData = require('minecraft-data')(this.bot.version);
                const biomeData = mcData.biomes[biomeId];
                if (biomeData?.name) return biomeData.name;
                if (biomeData?.displayName) return biomeData.displayName;
            }
            return 'unknown';
        } catch {
            return 'unknown';
        }
    }

    _detectNearbyStructures() {
        const structures = [];
        const pos = this.bot.entity.position;
        const reg = this.bot.registry.blocksByName;

        try {
            // Blacksmith: lava source + chest within 16 blocks
            const lavaId  = reg['lava']?.id;
            const chestId = reg['chest']?.id;
            if (lavaId && chestId) {
                const lava  = this.bot.findBlock({ matching: lavaId,  maxDistance: 16 });
                const chest = this.bot.findBlock({ matching: chestId, maxDistance: 16 });
                if (lava && chest) structures.push('blacksmith');
            }

            // Village: villager entity within 48 blocks
            const villager = Object.values(this.bot.entities).find(
                e => e.name === 'villager' &&
                     e.position.distanceTo(pos) <= 48
            );
            if (villager) structures.push('village');

            // Desert temple: orange terracotta within 32 blocks
            const orangeId = reg['orange_terracotta']?.id;
            if (orangeId) {
                const ot = this.bot.findBlock({ matching: orangeId, maxDistance: 32 });
                if (ot) structures.push('desert_temple');
            }

            // Jungle temple: mossy_cobblestone in jungle-ish area within 32 blocks
            const mossyId = reg['mossy_cobblestone']?.id;
            if (mossyId) {
                const mc = this.bot.findBlock({ matching: mossyId, maxDistance: 32 });
                if (mc) {
                    // mossy_cobblestone also signals dungeon — distinguish by spawner
                    const spawnerId = reg['spawner']?.id;
                    if (spawnerId) {
                        const spawner = this.bot.findBlock({ matching: spawnerId, maxDistance: 20 });
                        if (spawner) {
                            structures.push('dungeon');
                        } else {
                            structures.push('jungle_temple');
                        }
                    } else {
                        structures.push('jungle_temple');
                    }
                }
            }

            // Mineshaft: oak_fence underground (y < 40) within 32 blocks
            const fenceId = reg['oak_fence']?.id;
            if (fenceId && pos.y < 40) {
                const fence = this.bot.findBlock({ matching: fenceId, maxDistance: 32 });
                if (fence) structures.push('mineshaft');
            }

            // Ruined portal: crying_obsidian within 32 blocks
            const cryingId = reg['crying_obsidian']?.id;
            if (cryingId) {
                const co = this.bot.findBlock({ matching: cryingId, maxDistance: 32 });
                if (co) structures.push('ruined_portal');
            }

            // Igloo: snow_block + white_wool within 16 blocks
            const snowId = reg['snow_block']?.id;
            const woolId = reg['white_wool']?.id;
            if (snowId && woolId) {
                const snow = this.bot.findBlock({ matching: snowId, maxDistance: 16 });
                const wool = this.bot.findBlock({ matching: woolId, maxDistance: 16 });
                if (snow && wool) structures.push('igloo');
            }

            // Shipwreck: spruce_planks in water / at low y
            const spruceId = reg['spruce_planks']?.id;
            if (spruceId && pos.y <= 64) {
                const sp = this.bot.findBlock({ matching: spruceId, maxDistance: 32 });
                if (sp && sp.y <= 62) structures.push('shipwreck');
            }

        } catch (e) {
            // Structure detection is best-effort — never crash the state builder
        }

        return structures.length > 0 ? structures : ['none'];
    }

    _checkDone() {
        // Episode ends if:
        // 1. Bot dies (health <= 0)
        // 2. Reached a goal state (e.g., has diamond pickaxe)
        // 3. Max steps exceeded (handled separately as truncated)
        
        if (this.bot.health <= 0) {
            return true;
        }
        
        // Goal: Full diamond kit (pickaxe + all 4 armor pieces)
        const inv = this._getInventory();
        const fullDiamondKit = [
            'diamond_pickaxe',
            'diamond_helmet',
            'diamond_chestplate',
            'diamond_leggings',
            'diamond_boots'
        ];
        if (fullDiamondKit.every(item => inv[item])) {
            return true;
        }
        
        return false;
    }

    /**
     * Stop the WebSocket server
     */
    stop() {
        if (this._stuckInterval) {
            clearInterval(this._stuckInterval);
            this._stuckInterval = null;
        }
        if (this.wss) {
            this.wss.close();
            console.log('[Bridge] WebSocket server stopped');
        }
    }

    // ─────────────────────────────────────────────
    //  STUCK DETECTION & ESCAPE
    // ─────────────────────────────────────────────

    _resetStuckState() {
        this._lastPos      = null;
        this._lastMoveTime = Date.now();
        this._stuckLevel   = 0;
    }

    _startStuckDetector() {
        this._resetStuckState();

        // Also reset when the bot respawns after a kill
        this.bot.on('spawn', () => {
            console.log('[StuckDetector] Bot spawned/respawned — resetting stuck state');
            this._resetStuckState();
        });

        this._stuckInterval = setInterval(() => this._checkStuck(), STUCK_CHECK_INTERVAL_MS);
        console.log('[StuckDetector] Started');
    }

    async _checkStuck() {
        if (!this.bot?.entity) return;

        const pos  = this.bot.entity.position;
        const now  = Date.now();

        if (!this._lastPos) {
            this._lastPos = pos.clone();
            return;
        }

        const moved = pos.distanceTo(this._lastPos);

        if (moved > STUCK_MOVE_THRESHOLD) {
            // Bot has moved — update position and timer.
            // Only reset the escape level if the bot moved substantially (> 2 blocks),
            // not on a tiny hop from L1's jump+walk.  This prevents the infinite L1 loop
            // where a small jump resets _stuckLevel to 0 before the ladder can escalate.
            this._lastPos      = pos.clone();
            this._lastMoveTime = now;
            if (moved > 2.0) this._stuckLevel = 0;
            return;
        }

        const stuckMs = now - this._lastMoveTime;

        // L1 and L2 call pathfinder.stop() and bot.dig() which directly abort
        // whatever skill is currently executing, causing spurious "Path was stopped"
        // and "Digging aborted" failures.  Skip them while a skill is in flight;
        // only allow L3/L4 (which ultimately kill/respawn the bot) through.
        const skillRunning = this._skillRunning;

        if (stuckMs >= STUCK_L4_MS && this._stuckLevel < 4) {
            this._stuckLevel = 4;
            console.warn(`[StuckDetector] L4 (${stuckMs / 1000}s stuck) — RCON kill`);
            await this._rconKill();
            this._lastMoveTime = now; // give it time to respawn before next check

        } else if (stuckMs >= STUCK_L3_MS && this._stuckLevel < 3 && !skillRunning) {
            this._stuckLevel = 3;
            console.warn(`[StuckDetector] L3 (${stuckMs / 1000}s stuck) — place block + jump`);
            await this._escapePlaceBlock();

        } else if (stuckMs >= STUCK_L2_MS && this._stuckLevel < 2 && !skillRunning) {
            this._stuckLevel = 2;
            console.warn(`[StuckDetector] L2 (${stuckMs / 1000}s stuck) — mine surrounding blocks`);
            await this._escapeMineAround();

        } else if (stuckMs >= STUCK_L1_MS && this._stuckLevel < 1 && !skillRunning) {
            this._stuckLevel = 1;
            console.warn(`[StuckDetector] L1 (${stuckMs / 1000}s stuck) — jump + walk`);
            await this._escapeJumpWalk();
        }
    }

    /** L1: stop pathfinder, jump 3×, walk in a random direction for 1 s */
    async _escapeJumpWalk() {
        try {
            this.bot.pathfinder.stop();
        } catch (_) {}

        try {
            for (let i = 0; i < 3; i++) {
                this.bot.setControlState('jump', true);
                await this._wait(300);
                this.bot.setControlState('jump', false);
                await this._wait(200);
            }
            const dirs = ['forward', 'back', 'left', 'right'];
            const dir  = dirs[Math.floor(Math.random() * dirs.length)];
            this.bot.setControlState(dir, true);
            await this._wait(1000);
            this.bot.setControlState(dir, false);
        } catch (e) {
            console.warn('[StuckDetector] L1 escape error:', e.message);
        }
    }

    /** L2: mine every non-air block within 1 block in all 6 directions */
    async _escapeMineAround() {
        try {
            this.bot.pathfinder.stop();
        } catch (_) {}

        const pos = this.bot.entity.position;
        const offsets = [
            [0, 0, 0], [0, 1, 0], [0, -1, 0],
            [1, 0, 0], [-1, 0, 0], [0, 0, 1], [0, 0, -1],
            [1, 1, 0], [-1, 1, 0], [0, 1, 1], [0, 1, -1]
        ];

        for (const [dx, dy, dz] of offsets) {
            try {
                const block = this.bot.blockAt(pos.offset(dx, dy, dz));
                if (block && !['air', 'water', 'lava'].includes(block.name)) {
                    await this.bot.dig(block);
                }
            } catch (_) {}
        }

        // Jump after clearing space
        this.bot.setControlState('jump', true);
        await this._wait(400);
        this.bot.setControlState('jump', false);
    }

    /** L3: place a dirt/cobblestone block beneath the bot (if floating), then jump */
    async _escapePlaceBlock() {
        try {
            this.bot.pathfinder.stop();
        } catch (_) {}

        try {
            // Find a placeable block in inventory (dirt, cobblestone, any solid)
            const placeable = this.bot.inventory.items().find(i =>
                ['dirt', 'cobblestone', 'cobbled_deepslate', 'gravel',
                 'stone', 'sand', 'oak_planks'].includes(i.name)
            );

            if (placeable) {
                const blockBelow = this.bot.blockAt(this.bot.entity.position.offset(0, -1, 0));
                if (blockBelow && blockBelow.name === 'air') {
                    // Bot is floating — place block beneath
                    const ref = this.bot.blockAt(this.bot.entity.position.offset(0, -2, 0));
                    if (ref) {
                        await this.bot.equip(placeable, 'hand');
                        await this.bot.placeBlock(ref, new (require('vec3'))(0, 1, 0));
                    }
                }
            }
        } catch (_) {}

        // Jump regardless
        this.bot.setControlState('jump', true);
        await this._wait(500);
        this.bot.setControlState('jump', false);
    }

    /**
     * L4: Kill the bot via RCON so it respawns at its spawn point.
     * Uses the raw RCON TCP protocol — no extra npm packages needed.
     */
    async _rconKill() {
        const killed = await this._rconCommand(`kill ${this.bot.username}`);
        if (killed) {
            console.log('[StuckDetector] Bot killed via RCON — waiting for respawn');
        } else {
            // RCON unavailable — try chat command as last-ditch (works if bot is op'd)
            console.warn('[StuckDetector] RCON failed — trying /kill via chat');
            try { this.bot.chat('/kill'); } catch (_) {}
        }
    }

    /**
     * Send a command to the server via raw RCON TCP.
     * Packet format: [int32 length][int32 requestId][int32 type][payload\0\0]
     */
    _rconCommand(command) {
        return new Promise(resolve => {
            const net = require('net');
            const client = new net.Socket();
            let authed = false;
            let buf = Buffer.alloc(0);

            const send = (type, id, payload) => {
                const body = Buffer.from(payload + '\x00\x00', 'utf8');
                const pkt  = Buffer.alloc(4 + 4 + 4 + body.length);
                pkt.writeInt32LE(8 + body.length, 0);
                pkt.writeInt32LE(id, 4);
                pkt.writeInt32LE(type, 8);
                body.copy(pkt, 12);
                client.write(pkt);
            };

            const cleanup = (ok) => { try { client.destroy(); } catch (_) {} resolve(ok); };

            setTimeout(() => cleanup(false), 5000);

            client.connect(RCON_PORT, '127.0.0.1', () => send(3, 1, RCON_PASSWORD));

            client.on('data', data => {
                buf = Buffer.concat([buf, data]);
                while (buf.length >= 12) {
                    const len = buf.readInt32LE(0);
                    if (buf.length < len + 4) break;
                    buf = buf.slice(len + 4);
                    if (!authed) {
                        authed = true;
                        send(2, 2, command);   // type 2 = run command
                    } else {
                        cleanup(true);
                    }
                }
            });

            client.on('error', () => cleanup(false));
        });
    }

    _wait(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}

module.exports = Bridge;
