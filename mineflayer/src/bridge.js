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
            'stone', 'cobblestone', 'iron_ore', 'coal_ore', 'diamond_ore',
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
            return block?.biome?.name || 'unknown';
        } catch {
            return 'unknown';
        }
    }

    _checkDone() {
        // Episode ends if:
        // 1. Bot dies (health <= 0)
        // 2. Reached a goal state (e.g., has diamond pickaxe)
        // 3. Max steps exceeded (handled separately as truncated)
        
        if (this.bot.health <= 0) {
            return true;
        }
        
        // Goal: Crafted iron pickaxe (a significant milestone)
        if (this._getInventory()['iron_pickaxe']) {
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
