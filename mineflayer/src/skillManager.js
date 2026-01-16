/**
 * SkillManager - Manages the Skill Library for Hierarchical RL
 * 
 * Following the Options Framework, each skill is a "temporally extended action"
 * with its own initiation set, termination condition, and internal policy.
 * 
 * Inspired by:
 * - Voyager: Code-as-action paradigm
 * - Plan4MC: Basic skills as atomic units for graph search
 * - DEPS: Error handling and retry logic
 */

const { goals, Movements } = require('mineflayer-pathfinder');
const { GoalNear, GoalBlock, GoalGetToBlock } = goals;

class SkillManager {
    constructor(bot) {
        this.bot = bot;
        this.currentSkill = null;
        this.skillRegistry = new Map();
        this.executionHistory = [];
        
        // Register all available skills
        this._registerSkills();
    }

    /**
     * Register all primitive skills in the library
     * Each skill follows the Options Framework structure:
     * - id: Unique identifier (action space index)
     * - name: Human-readable name
     * - preconditions: Function checking if skill can be initiated
     * - execute: The skill's internal policy
     * - termination: Conditions for skill completion
     */
    _registerSkills() {
        // Skill 0: Idle / No-op
        this.register({
            id: 0,
            name: 'idle',
            description: 'Do nothing for one tick',
            preconditions: () => true,
            execute: async () => {
                await this._wait(50);
                return { success: true, message: 'Idle complete' };
            }
        });

        // Skill 1: Harvest nearest tree (punch wood)
        this.register({
            id: 1,
            name: 'harvest_wood',
            description: 'Find and harvest the nearest tree',
            preconditions: () => true, // Can always attempt
            execute: async () => {
                return await this._harvestWood();
            }
        });

        // Skill 2: Mine stone
        this.register({
            id: 2,
            name: 'mine_stone',
            description: 'Mine cobblestone (requires wooden pickaxe)',
            preconditions: () => this._hasItem('wooden_pickaxe') || this._hasItem('stone_pickaxe'),
            execute: async () => {
                return await this._mineBlock('stone', 3);
            }
        });

        // Skill 3: Craft planks
        this.register({
            id: 3,
            name: 'craft_planks',
            description: 'Craft wooden planks from logs',
            preconditions: () => this._hasItemLike('_log'),
            execute: async () => {
                return await this._craftItem('oak_planks', 1);
            }
        });

        // Skill 4: Craft sticks
        this.register({
            id: 4,
            name: 'craft_sticks',
            description: 'Craft sticks from planks',
            preconditions: () => this._hasItemLike('_planks'),
            execute: async () => {
                return await this._craftItem('stick', 1);
            }
        });

        // Skill 5: Craft crafting table
        this.register({
            id: 5,
            name: 'craft_crafting_table',
            description: 'Craft a crafting table',
            preconditions: () => this._countItemLike('_planks') >= 4,
            execute: async () => {
                return await this._craftItem('crafting_table', 1);
            }
        });

        // Skill 6: Craft wooden pickaxe
        this.register({
            id: 6,
            name: 'craft_wooden_pickaxe',
            description: 'Craft a wooden pickaxe (requires crafting table nearby)',
            preconditions: () => this._countItemLike('_planks') >= 3 && this._countItem('stick') >= 2,
            execute: async () => {
                return await this._craftWithTable('wooden_pickaxe', 1);
            }
        });

        // Skill 7: Craft stone pickaxe
        this.register({
            id: 7,
            name: 'craft_stone_pickaxe',
            description: 'Craft a stone pickaxe',
            preconditions: () => this._countItem('cobblestone') >= 3 && this._countItem('stick') >= 2,
            execute: async () => {
                return await this._craftWithTable('stone_pickaxe', 1);
            }
        });

        // Skill 8: Eat food
        this.register({
            id: 8,
            name: 'eat_food',
            description: 'Eat available food to restore hunger',
            preconditions: () => this._hasFood() && this.bot.food < 20,
            execute: async () => {
                return await this._eatFood();
            }
        });

        // Skill 9: Navigate to nearest structure (village, etc.)
        this.register({
            id: 9,
            name: 'explore',
            description: 'Explore in a random direction',
            preconditions: () => true,
            execute: async () => {
                return await this._explore();
            }
        });

        // Skill 10: Place crafting table
        this.register({
            id: 10,
            name: 'place_crafting_table',
            description: 'Place a crafting table nearby',
            preconditions: () => this._hasItem('crafting_table'),
            execute: async () => {
                return await this._placeBlock('crafting_table');
            }
        });

        // Skill 11: Mine iron ore
        this.register({
            id: 11,
            name: 'mine_iron',
            description: 'Mine iron ore (requires stone pickaxe)',
            preconditions: () => this._hasItem('stone_pickaxe') || this._hasItem('iron_pickaxe'),
            execute: async () => {
                return await this._mineBlock('iron_ore', 3);
            }
        });

        // Skill 12: Smelt iron (requires furnace)
        this.register({
            id: 12,
            name: 'smelt_iron',
            description: 'Smelt raw iron into iron ingots',
            preconditions: () => this._hasItem('raw_iron') && this._hasItem('furnace'),
            execute: async () => {
                return await this._smeltItem('raw_iron', 'iron_ingot');
            }
        });
    }

    /**
     * Register a new skill
     */
    register(skill) {
        this.skillRegistry.set(skill.id, skill);
        console.log(`[SkillManager] Registered skill ${skill.id}: ${skill.name}`);
    }

    /**
     * Get all available skills (for action space definition)
     */
    getSkillList() {
        return Array.from(this.skillRegistry.values()).map(s => ({
            id: s.id,
            name: s.name,
            description: s.description,
            available: s.preconditions()
        }));
    }

    /**
     * Get number of skills (action space size)
     */
    getActionSpaceSize() {
        return this.skillRegistry.size;
    }

    /**
     * Execute a skill by ID
     * Returns: { success: bool, message: string, reward: number }
     */
    async executeSkill(skillId) {
        const skill = this.skillRegistry.get(skillId);
        
        if (!skill) {
            return { 
                success: false, 
                message: `Unknown skill ID: ${skillId}`,
                reward: -1 
            };
        }

        // Check preconditions (initiation set in Options Framework)
        if (!skill.preconditions()) {
            return { 
                success: false, 
                message: `Preconditions not met for ${skill.name}`,
                reward: -0.5 
            };
        }

        console.log(`[SkillManager] Executing skill: ${skill.name}`);
        this.currentSkill = skill;
        
        const startTime = Date.now();
        const startInventory = this._getInventorySnapshot();
        
        try {
            const result = await skill.execute();
            const endInventory = this._getInventorySnapshot();
            
            // Calculate intrinsic reward based on inventory changes
            const inventoryReward = this._calculateInventoryReward(startInventory, endInventory);
            
            this.executionHistory.push({
                skillId,
                skillName: skill.name,
                success: result.success,
                duration: Date.now() - startTime,
                timestamp: Date.now()
            });

            return {
                ...result,
                reward: result.success ? (0.1 + inventoryReward) : -0.1
            };
        } catch (error) {
            console.error(`[SkillManager] Skill ${skill.name} failed:`, error.message);
            return { 
                success: false, 
                message: error.message,
                reward: -0.5 
            };
        } finally {
            this.currentSkill = null;
        }
    }

    // ==================== Skill Implementations ====================

    async _harvestWood() {
        const logTypes = ['oak_log', 'birch_log', 'spruce_log', 'jungle_log', 'acacia_log', 'dark_oak_log'];
        
        for (const logType of logTypes) {
            const log = this.bot.findBlock({
                matching: block => block.name === logType,
                maxDistance: 64
            });
            
            if (log) {
                await this._goTo(log.position);
                await this.bot.dig(log);
                return { success: true, message: `Harvested ${logType}` };
            }
        }
        
        return { success: false, message: 'No trees found nearby' };
    }

    async _mineBlock(blockName, count = 1) {
        let mined = 0;
        
        while (mined < count) {
            const block = this.bot.findBlock({
                matching: b => b.name === blockName || b.name.includes(blockName),
                maxDistance: 32
            });
            
            if (!block) {
                return { 
                    success: mined > 0, 
                    message: `Mined ${mined}/${count} ${blockName}` 
                };
            }
            
            // Equip best tool
            await this._equipBestTool(block);
            await this._goTo(block.position);
            await this.bot.dig(block);
            mined++;
        }
        
        return { success: true, message: `Mined ${count} ${blockName}` };
    }

    async _craftItem(itemName, count = 1) {
        const mcData = require('minecraft-data')(this.bot.version);
        const recipe = this.bot.recipesFor(mcData.itemsByName[itemName]?.id)[0];
        
        if (!recipe) {
            return { success: false, message: `No recipe found for ${itemName}` };
        }
        
        try {
            await this.bot.craft(recipe, count, null);
            return { success: true, message: `Crafted ${count} ${itemName}` };
        } catch (error) {
            return { success: false, message: `Failed to craft ${itemName}: ${error.message}` };
        }
    }

    async _craftWithTable(itemName, count = 1) {
        const mcData = require('minecraft-data')(this.bot.version);
        
        // Find or place crafting table
        let craftingTable = this.bot.findBlock({
            matching: b => b.name === 'crafting_table',
            maxDistance: 32
        });
        
        if (!craftingTable && this._hasItem('crafting_table')) {
            const placeResult = await this._placeBlock('crafting_table');
            if (!placeResult.success) return placeResult;
            
            craftingTable = this.bot.findBlock({
                matching: b => b.name === 'crafting_table',
                maxDistance: 8
            });
        }
        
        if (!craftingTable) {
            return { success: false, message: 'No crafting table available' };
        }
        
        await this._goTo(craftingTable.position);
        
        const recipe = this.bot.recipesFor(mcData.itemsByName[itemName]?.id, null, 1, craftingTable)[0];
        
        if (!recipe) {
            return { success: false, message: `No recipe found for ${itemName}` };
        }
        
        try {
            await this.bot.craft(recipe, count, craftingTable);
            return { success: true, message: `Crafted ${count} ${itemName}` };
        } catch (error) {
            return { success: false, message: `Failed to craft ${itemName}: ${error.message}` };
        }
    }

    async _placeBlock(itemName) {
        const item = this.bot.inventory.items().find(i => i.name === itemName);
        if (!item) {
            return { success: false, message: `No ${itemName} in inventory` };
        }
        
        // Find a suitable position to place
        const pos = this.bot.entity.position.offset(1, 0, 0).floored();
        const referenceBlock = this.bot.blockAt(pos.offset(0, -1, 0));
        
        if (!referenceBlock) {
            return { success: false, message: 'No reference block found' };
        }
        
        try {
            await this.bot.equip(item, 'hand');
            await this.bot.placeBlock(referenceBlock, new (require('vec3'))(0, 1, 0));
            return { success: true, message: `Placed ${itemName}` };
        } catch (error) {
            return { success: false, message: `Failed to place ${itemName}: ${error.message}` };
        }
    }

    async _eatFood() {
        const foodItems = this.bot.inventory.items().filter(item => 
            item.name.includes('apple') || 
            item.name.includes('bread') || 
            item.name.includes('cooked') ||
            item.name.includes('steak') ||
            item.name.includes('porkchop')
        );
        
        if (foodItems.length === 0) {
            return { success: false, message: 'No food available' };
        }
        
        try {
            await this.bot.equip(foodItems[0], 'hand');
            await this.bot.consume();
            return { success: true, message: `Ate ${foodItems[0].name}` };
        } catch (error) {
            return { success: false, message: `Failed to eat: ${error.message}` };
        }
    }

    async _explore() {
        const { pathfinder, Movements } = require('mineflayer-pathfinder');
        const mcData = require('minecraft-data')(this.bot.version);
        
        // Random direction
        const angle = Math.random() * 2 * Math.PI;
        const distance = 30 + Math.random() * 50;
        
        const targetX = this.bot.entity.position.x + Math.cos(angle) * distance;
        const targetZ = this.bot.entity.position.z + Math.sin(angle) * distance;
        
        const goal = new GoalNear(targetX, this.bot.entity.position.y, targetZ, 5);
        
        try {
            const defaultMove = new Movements(this.bot, mcData);
            this.bot.pathfinder.setMovements(defaultMove);
            await this.bot.pathfinder.goto(goal);
            return { success: true, message: `Explored to ${targetX.toFixed(0)}, ${targetZ.toFixed(0)}` };
        } catch (error) {
            return { success: false, message: `Exploration failed: ${error.message}` };
        }
    }

    async _smeltItem(inputItem, outputItem) {
        // Placeholder - requires furnace interaction
        return { success: false, message: 'Smelting not yet implemented' };
    }

    // ==================== Helper Methods ====================

    async _goTo(position) {
        const { pathfinder, Movements, goals } = require('mineflayer-pathfinder');
        const mcData = require('minecraft-data')(this.bot.version);
        
        const defaultMove = new Movements(this.bot, mcData);
        this.bot.pathfinder.setMovements(defaultMove);
        
        const goal = new goals.GoalNear(position.x, position.y, position.z, 2);
        await this.bot.pathfinder.goto(goal);
    }

    async _equipBestTool(block) {
        try {
            await this.bot.tool.equipForBlock(block);
        } catch (error) {
            // Ignore - will use hand if no tool available
        }
    }

    _wait(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }

    _hasItem(itemName) {
        return this.bot.inventory.items().some(i => i.name === itemName);
    }

    _hasItemLike(pattern) {
        return this.bot.inventory.items().some(i => i.name.includes(pattern));
    }

    _countItem(itemName) {
        return this.bot.inventory.items()
            .filter(i => i.name === itemName)
            .reduce((sum, i) => sum + i.count, 0);
    }

    _countItemLike(pattern) {
        return this.bot.inventory.items()
            .filter(i => i.name.includes(pattern))
            .reduce((sum, i) => sum + i.count, 0);
    }

    _hasFood() {
        return this.bot.inventory.items().some(item => 
            item.name.includes('apple') || 
            item.name.includes('bread') || 
            item.name.includes('cooked') ||
            item.name.includes('steak')
        );
    }

    _getInventorySnapshot() {
        const snapshot = {};
        for (const item of this.bot.inventory.items()) {
            snapshot[item.name] = (snapshot[item.name] || 0) + item.count;
        }
        return snapshot;
    }

    _calculateInventoryReward(before, after) {
        // Reward for gaining new items
        let reward = 0;
        const techTreeValues = {
            'oak_log': 0.1, 'birch_log': 0.1, 'spruce_log': 0.1,
            'oak_planks': 0.05, 'birch_planks': 0.05,
            'stick': 0.02,
            'crafting_table': 0.3,
            'wooden_pickaxe': 0.5,
            'cobblestone': 0.1,
            'stone_pickaxe': 1.0,
            'iron_ore': 0.5,
            'raw_iron': 0.5,
            'iron_ingot': 1.0,
            'iron_pickaxe': 2.0
        };
        
        for (const [item, count] of Object.entries(after)) {
            const gained = count - (before[item] || 0);
            if (gained > 0 && techTreeValues[item]) {
                reward += techTreeValues[item] * gained;
            }
        }
        
        return reward;
    }
}

module.exports = SkillManager;
