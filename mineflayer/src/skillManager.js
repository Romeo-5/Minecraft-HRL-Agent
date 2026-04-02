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
const { GoalNear, GoalBlock, GoalGetToBlock, GoalY } = goals;

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
            // Stop crafting once we have a comfortable surplus (32 planks = 8 logs worth)
            preconditions: () => this._hasItemLike('_log') && this._countItemLike('_planks') < 32,
            execute: async () => {
                return await this._craftItem('oak_planks', 1);
            }
        });

        // Skill 4: Craft sticks
        this.register({
            id: 4,
            name: 'craft_sticks',
            description: 'Craft sticks from planks',
            // 16 sticks covers 5+ pickaxes — no reason to hoard more
            preconditions: () => this._hasItemLike('_planks') && this._countItem('stick') < 16,
            execute: async () => {
                return await this._craftItem('stick', 1);
            }
        });

        // Skill 5: Craft crafting table
        this.register({
            id: 5,
            name: 'craft_crafting_table',
            description: 'Craft a crafting table',
            // One crafting table is enough — check inventory AND nearby world
            preconditions: () => {
                if (this._countItemLike('_planks') < 4) return false;
                if (this._hasItem('crafting_table')) return false; // already carrying one
                const placed = this.bot.findBlock({ matching: b => b.name === 'crafting_table', maxDistance: 32 });
                return placed === null; // only craft if none placed nearby
            },
            execute: async () => {
                return await this._craftItem('crafting_table', 1);
            }
        });

        // Skill 6: Craft wooden pickaxe
        this.register({
            id: 6,
            name: 'craft_wooden_pickaxe',
            description: 'Craft a wooden pickaxe (requires crafting table nearby)',
            // Only craft if we don't already have any pickaxe (wood or better)
            preconditions: () =>
                this._countItemLike('_planks') >= 3 &&
                this._countItem('stick') >= 2 &&
                this._countItem('wooden_pickaxe') < 1 &&
                !this._hasItem('stone_pickaxe') &&
                !this._hasItem('iron_pickaxe') &&
                !this._hasItem('diamond_pickaxe'),
            execute: async () => {
                return await this._craftWithTable('wooden_pickaxe', 1);
            }
        });

        // Skill 7: Craft stone pickaxe
        this.register({
            id: 7,
            name: 'craft_stone_pickaxe',
            description: 'Craft a stone pickaxe',
            // Only craft if we don't already have a stone or better pickaxe
            preconditions: () =>
                this._countItem('cobblestone') >= 3 &&
                this._countItem('stick') >= 2 &&
                this._countItem('stone_pickaxe') < 1 &&
                !this._hasItem('iron_pickaxe') &&
                !this._hasItem('diamond_pickaxe'),
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

        // Skill 12: Smelt iron (finds/crafts/places furnace automatically)
        this.register({
            id: 12,
            name: 'smelt_iron',
            description: 'Smelt raw iron into iron ingots',
            preconditions: () => {
                if (!this._hasItem('raw_iron')) return false;
                // OK if we have a furnace item, enough cobble to make one, or one already in world
                if (this._hasItem('furnace')) return true;
                if (this._countItem('cobblestone') >= 8) return true;
                const nearby = this.bot.findBlock({ matching: b => b.name === 'furnace', maxDistance: 32 });
                return nearby !== null;
            },
            execute: async () => {
                return await this._smeltItem('raw_iron', 'iron_ingot');
            }
        });

        // Skill 13: Craft furnace
        this.register({
            id: 13,
            name: 'craft_furnace',
            description: 'Craft a furnace from 8 cobblestone',
            // One furnace is enough — smelt_iron will auto-craft/place one anyway
            preconditions: () =>
                this._countItem('cobblestone') >= 8 &&
                this._countItem('furnace') < 1 &&
                this.bot.findBlock({ matching: b => b.name === 'furnace', maxDistance: 32 }) === null,
            execute: async () => {
                return await this._craftWithTable('furnace', 1);
            }
        });

        // Skill 14: Craft iron pickaxe
        this.register({
            id: 14,
            name: 'craft_iron_pickaxe',
            description: 'Craft an iron pickaxe (requires 3 iron ingots + 2 sticks + crafting table)',
            // Only craft if we don't already have an iron or diamond pickaxe
            preconditions: () =>
                this._countItem('iron_ingot') >= 3 &&
                this._countItem('stick') >= 2 &&
                this._countItem('iron_pickaxe') < 1 &&
                !this._hasItem('diamond_pickaxe'),
            execute: async () => {
                return await this._craftWithTable('iron_pickaxe', 1);
            }
        });

        // Skill 15: Craft iron helmet
        this.register({
            id: 15,
            name: 'craft_iron_helmet',
            description: 'Craft an iron helmet (5 iron ingots + crafting table)',
            preconditions: () =>
                this._countItem('iron_ingot') >= 5 &&
                this._countItem('iron_helmet') < 1,
            execute: async () => {
                return await this._craftWithTable('iron_helmet', 1);
            }
        });

        // Skill 16: Craft iron chestplate
        this.register({
            id: 16,
            name: 'craft_iron_chestplate',
            description: 'Craft an iron chestplate (8 iron ingots + crafting table)',
            preconditions: () =>
                this._countItem('iron_ingot') >= 8 &&
                this._countItem('iron_chestplate') < 1,
            execute: async () => {
                return await this._craftWithTable('iron_chestplate', 1);
            }
        });

        // Skill 17: Craft iron leggings
        this.register({
            id: 17,
            name: 'craft_iron_leggings',
            description: 'Craft iron leggings (7 iron ingots + crafting table)',
            preconditions: () =>
                this._countItem('iron_ingot') >= 7 &&
                this._countItem('iron_leggings') < 1,
            execute: async () => {
                return await this._craftWithTable('iron_leggings', 1);
            }
        });

        // Skill 18: Craft iron boots
        this.register({
            id: 18,
            name: 'craft_iron_boots',
            description: 'Craft iron boots (4 iron ingots + crafting table)',
            preconditions: () =>
                this._countItem('iron_ingot') >= 4 &&
                this._countItem('iron_boots') < 1,
            execute: async () => {
                return await this._craftWithTable('iron_boots', 1);
            }
        });

        // Skill 19: Dig to diamond level
        this.register({
            id: 19,
            name: 'dig_to_diamond_level',
            description: 'Dig down to Y=-59 (diamond ore level), requires iron pickaxe',
            preconditions: () => this._hasItem('iron_pickaxe') && Math.floor(this.bot.entity.position.y) > -50,
            execute: async () => {
                return await this._digToY(-59);
            }
        });

        // Skill 20: Return to surface
        this.register({
            id: 20,
            name: 'return_to_surface',
            description: 'Navigate from underground back to the surface (Y>=60)',
            preconditions: () => Math.floor(this.bot.entity.position.y) < 0,
            execute: async () => {
                return await this._returnToSurface();
            }
        });

        // Skill 21: Mine diamond ore
        this.register({
            id: 21,
            name: 'mine_diamond',
            description: 'Mine deepslate diamond ore (requires iron pickaxe, must be at Y<=-50)',
            preconditions: () => this._hasItem('iron_pickaxe') && Math.floor(this.bot.entity.position.y) <= -50,
            execute: async () => {
                return await this._mineBlock('diamond_ore', 3);
            }
        });

        // Skill 22: Craft diamond pickaxe
        this.register({
            id: 22,
            name: 'craft_diamond_pickaxe',
            description: 'Craft a diamond pickaxe (3 diamonds + 2 sticks + crafting table)',
            preconditions: () =>
                this._countItem('diamond') >= 3 &&
                this._countItem('stick') >= 2 &&
                this._countItem('diamond_pickaxe') < 1,
            execute: async () => {
                return await this._craftWithTable('diamond_pickaxe', 1);
            }
        });

        // Skill 23: Craft diamond helmet
        this.register({
            id: 23,
            name: 'craft_diamond_helmet',
            description: 'Craft a diamond helmet (5 diamonds + crafting table)',
            preconditions: () =>
                this._countItem('diamond') >= 5 &&
                this._countItem('diamond_helmet') < 1,
            execute: async () => {
                return await this._craftWithTable('diamond_helmet', 1);
            }
        });

        // Skill 24: Craft diamond chestplate
        this.register({
            id: 24,
            name: 'craft_diamond_chestplate',
            description: 'Craft a diamond chestplate (8 diamonds + crafting table)',
            preconditions: () =>
                this._countItem('diamond') >= 8 &&
                this._countItem('diamond_chestplate') < 1,
            execute: async () => {
                return await this._craftWithTable('diamond_chestplate', 1);
            }
        });

        // Skill 25: Craft diamond leggings
        this.register({
            id: 25,
            name: 'craft_diamond_leggings',
            description: 'Craft diamond leggings (7 diamonds + crafting table)',
            preconditions: () =>
                this._countItem('diamond') >= 7 &&
                this._countItem('diamond_leggings') < 1,
            execute: async () => {
                return await this._craftWithTable('diamond_leggings', 1);
            }
        });

        // Skill 26: Craft diamond boots
        this.register({
            id: 26,
            name: 'craft_diamond_boots',
            description: 'Craft diamond boots (4 diamonds + crafting table)',
            preconditions: () =>
                this._countItem('diamond') >= 4 &&
                this._countItem('diamond_boots') < 1,
            execute: async () => {
                return await this._craftWithTable('diamond_boots', 1);
            }
        });

        // Skill 27: Clear junk inventory
        // Drops low-value blocks that clog inventory during deep mining.
        // Only fires when inventory is nearly full (>=27 of 36 slots occupied).
        this.register({
            id: 27,
            name: 'clear_junk',
            description: 'Drop low-value blocks to free inventory space (auto-triggered when nearly full)',
            preconditions: () => this.bot.inventory.items().length >= 27,
            execute: async () => {
                return await this._clearJunk();
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

        // Random direction, shorter distance to avoid wandering into ocean
        const angle = Math.random() * 2 * Math.PI;
        const distance = 15 + Math.random() * 20;

        const targetX = this.bot.entity.position.x + Math.cos(angle) * distance;
        const targetZ = this.bot.entity.position.z + Math.sin(angle) * distance;

        const goal = new GoalNear(targetX, this.bot.entity.position.y, targetZ, 5);

        try {
            const safeMove = new Movements(this.bot, mcData);
            safeMove.canSwim = false;          // never enter water voluntarily
            safeMove.allowSprinting = true;
            this.bot.pathfinder.setMovements(safeMove);
            await this.bot.pathfinder.goto(goal);
            return { success: true, message: `Explored to ${targetX.toFixed(0)}, ${targetZ.toFixed(0)}` };
        } catch (error) {
            return { success: false, message: `Exploration failed: ${error.message}` };
        }
    }

    async _clearJunk() {
        // Items worthless to the tech-tree that pile up during descent
        const junkNames = [
            'cobblestone', 'gravel', 'dirt', 'sand', 'andesite',
            'diorite', 'granite', 'tuff', 'calcite', 'deepslate',
            'cobbled_deepslate', 'stone', 'flint', 'clay_ball',
            'sandstone', 'netherrack'
        ];

        let dropped = 0;
        for (const item of this.bot.inventory.items()) {
            if (junkNames.some(j => item.name === j || item.name.includes(j))) {
                try {
                    await this.bot.toss(item.type, null, item.count);
                    dropped += item.count;
                } catch (_) { /* ignore toss errors */ }
            }
        }

        if (dropped === 0) {
            return { success: false, message: 'No junk items to drop' };
        }
        return { success: true, message: `Dropped ${dropped} junk items, freed inventory space` };
    }

    async _digToY(targetY) {
        const mcData = require('minecraft-data')(this.bot.version);

        const digMove = new Movements(this.bot, mcData);
        digMove.canDig = true;
        digMove.canSwim = false;
        digMove.allowSprinting = false; // slower = safer underground
        this.bot.pathfinder.setMovements(digMove);

        let lavaAbort = false;
        const lavaCheckInterval = setInterval(() => {
            for (let dy = 1; dy <= 5; dy++) {
                const checkBlock = this.bot.blockAt(this.bot.entity.position.offset(0, -dy, 0));
                if (checkBlock && checkBlock.name === 'lava') {
                    lavaAbort = true;
                    this.bot.pathfinder.stop();
                    clearInterval(lavaCheckInterval);
                    break;
                }
            }
        }, 2000);

        const timeoutPromise = new Promise((_, reject) =>
            setTimeout(() => reject(new Error('_digToY timed out after 3 minutes')), 180000)
        );

        try {
            await Promise.race([
                this.bot.pathfinder.goto(new GoalY(targetY)),
                timeoutPromise
            ]);
            if (lavaAbort) {
                return { success: false, message: 'Aborted: lava detected below' };
            }
            return { success: true, message: `Reached Y=${Math.floor(this.bot.entity.position.y)}` };
        } catch (error) {
            if (lavaAbort) {
                return { success: false, message: 'Aborted: lava detected below' };
            }
            return { success: false, message: `Dig to Y failed: ${error.message}` };
        } finally {
            clearInterval(lavaCheckInterval);
            // Always restore safe movement so other skills don't accidentally tunnel
            const safeMove = new Movements(this.bot, mcData);
            safeMove.canDig = false;
            safeMove.canSwim = false;
            this.bot.pathfinder.setMovements(safeMove);
        }
    }

    async _returnToSurface() {
        const mcData = require('minecraft-data')(this.bot.version);

        const climbMove = new Movements(this.bot, mcData);
        climbMove.canDig = true;   // may need to re-mine collapsed blocks
        climbMove.canSwim = false;
        climbMove.allowSprinting = false;
        this.bot.pathfinder.setMovements(climbMove);

        const timeoutPromise = new Promise((_, reject) =>
            setTimeout(() => reject(new Error('_returnToSurface timed out after 3 minutes')), 180000)
        );

        try {
            await Promise.race([
                this.bot.pathfinder.goto(new GoalY(64)),
                timeoutPromise
            ]);
            return { success: true, message: `Surfaced at Y=${Math.floor(this.bot.entity.position.y)}` };
        } catch (error) {
            return { success: false, message: `Return to surface failed: ${error.message}` };
        } finally {
            const safeMove = new Movements(this.bot, mcData);
            safeMove.canDig = false;
            safeMove.canSwim = false;
            this.bot.pathfinder.setMovements(safeMove);
        }
    }

    async _smeltItem(inputItem, outputItem) {
        // Step 1: Find an existing furnace in the world
        let furnaceBlock = this.bot.findBlock({
            matching: b => b.name === 'furnace',
            maxDistance: 32
        });

        // Step 2: If no furnace in world, craft one and place it
        if (!furnaceBlock) {
            if (!this._hasItem('furnace')) {
                // Craft from cobblestone
                const craftResult = await this._craftWithTable('furnace', 1);
                if (!craftResult.success) {
                    return { success: false, message: `Could not craft furnace: ${craftResult.message}` };
                }
            }
            // Place furnace next to bot
            const placeResult = await this._placeBlock('furnace');
            if (!placeResult.success) {
                return { success: false, message: `Could not place furnace: ${placeResult.message}` };
            }
            await this._wait(300); // let block appear in world
            furnaceBlock = this.bot.findBlock({
                matching: b => b.name === 'furnace',
                maxDistance: 8
            });
            if (!furnaceBlock) {
                return { success: false, message: 'Placed furnace but could not locate it' };
            }
        }

        // Step 3: Walk up to the furnace
        await this._goTo(furnaceBlock.position);

        // Step 4: Pick fuel — prefer coal/charcoal, fall back to planks/logs
        const fuelItem = this.bot.inventory.items().find(i =>
            i.name === 'coal' ||
            i.name === 'charcoal' ||
            i.name.includes('planks') ||
            i.name.includes('_log')
        );
        if (!fuelItem) {
            return { success: false, message: 'No fuel available (need coal, planks, or logs)' };
        }

        // Step 5: Check we still have the input item
        const inputObj = this.bot.inventory.items().find(i => i.name === inputItem);
        if (!inputObj) {
            return { success: false, message: `No ${inputItem} in inventory` };
        }

        try {
            const furnace = await this.bot.openFurnace(furnaceBlock);

            // Load fuel and input (smelt up to 8 at once)
            await furnace.putFuel(fuelItem.type, null, Math.min(fuelItem.count, 8));
            await furnace.putInput(inputObj.type, null, Math.min(inputObj.count, 8));

            // Wait for at least one output (each item takes ~10 game ticks = ~0.5 s real time)
            await new Promise((resolve, reject) => {
                const deadline = setTimeout(() => {
                    reject(new Error('Smelting timed out after 45 s'));
                }, 45000);

                const poll = setInterval(() => {
                    if (furnace.outputItem()) {
                        clearInterval(poll);
                        clearTimeout(deadline);
                        resolve();
                    }
                }, 500);
            });

            // Collect finished ingots
            await furnace.takeOutput();
            furnace.close();

            return { success: true, message: `Smelted ${inputItem} → ${outputItem}` };
        } catch (error) {
            return { success: false, message: `Smelting failed: ${error.message}` };
        }
    }

    // ==================== Helper Methods ====================

    async _goTo(position) {
        const { pathfinder, Movements, goals } = require('mineflayer-pathfinder');
        const mcData = require('minecraft-data')(this.bot.version);

        const safeMove = new Movements(this.bot, mcData);
        safeMove.canSwim = false;   // never route through water
        this.bot.pathfinder.setMovements(safeMove);

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
            // Wood tier
            'oak_log': 0.1, 'birch_log': 0.1, 'spruce_log': 0.1,
            'oak_planks': 0.05, 'birch_planks': 0.05,
            'stick': 0.02,
            'crafting_table': 0.3,
            'wooden_pickaxe': 0.5,
            // Stone tier
            'cobblestone': 0.1,
            'stone_pickaxe': 1.0,
            // Iron tier
            'coal': 0.1,
            'iron_ore': 0.5,
            'raw_iron': 0.5,
            'furnace': 0.8,
            'iron_ingot': 1.0,
            'iron_pickaxe': 2.0,
            'iron_helmet': 3.0,
            'iron_chestplate': 4.5,
            'iron_leggings': 4.0,
            'iron_boots': 2.5,
            // Diamond tier
            'diamond': 5.0,
            'diamond_pickaxe': 10.0,
            'diamond_helmet': 8.0,
            'diamond_chestplate': 12.0,
            'diamond_leggings': 10.0,
            'diamond_boots': 6.0
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
