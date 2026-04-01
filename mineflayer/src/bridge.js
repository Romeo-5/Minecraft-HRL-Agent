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

class Bridge {
    constructor(bot, skillManager, port = 8765) {
        this.bot = bot;
        this.skillManager = skillManager;
        this.port = port;
        this.wss = null;
        this.client = null;
        this.episodeSteps = 0;
        this.maxEpisodeSteps = 1000;
    }

    /**
     * Start the WebSocket server
     */
    start() {
        this.wss = new WebSocket.Server({ port: this.port });
        
        console.log(`[Bridge] WebSocket server started on port ${this.port}`);
        
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
        
        // Execute the skill
        const result = await this.skillManager.executeSkill(skillId);
        
        // Get new state
        const state = this._getState();
        
        // Check termination conditions
        const done = this._checkDone();
        
        // Build info dict (DEPS-style error feedback)
        const info = {
            skill_executed: skillId,
            skill_name: this.skillManager.skillRegistry.get(skillId)?.name || 'unknown',
            skill_success: result.success,
            skill_message: result.message,
            steps: this.episodeSteps,
            inventory_count: Object.keys(state.inventory).length
        };
        
        return {
            type: 'step_result',
            state: state,
            reward: result.reward,
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
        if (this.wss) {
            this.wss.close();
            console.log('[Bridge] WebSocket server stopped');
        }
    }
}

module.exports = Bridge;
