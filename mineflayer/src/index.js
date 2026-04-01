/**
 * Minecraft HRL Agent - Mineflayer Entry Point
 * 
 * This is the main entry point for the Minecraft bot. It:
 * 1. Creates and configures the Mineflayer bot
 * 2. Loads necessary plugins (pathfinder, tool, etc.)
 * 3. Initializes the SkillManager
 * 4. Starts the Bridge server for Python communication
 * 
 * Usage:
 *   node src/index.js [--host <host>] [--port <port>] [--username <username>]
 * 
 * Environment Variables:
 *   MC_HOST - Minecraft server host (default: localhost)
 *   MC_PORT - Minecraft server port (default: 25565)
 *   MC_USERNAME - Bot username (default: HRL_Agent)
 *   BRIDGE_PORT - WebSocket bridge port (default: 8765)
 *   VIEWER_PORT - Prismarine viewer HTTP port (default: 3007)
 */

const mineflayer = require('mineflayer');
const { pathfinder } = require('mineflayer-pathfinder');
const toolPlugin = require('mineflayer-tool').plugin;
const collectBlock = require('mineflayer-collectblock').plugin;
const { mineflayer: mineflayerViewer } = require('prismarine-viewer');

const SkillManager = require('./skillManager');
const Bridge = require('./bridge');

// Configuration
const config = {
    host: process.env.MC_HOST || 'localhost',
    port: parseInt(process.env.MC_PORT) || 25565,
    username: process.env.MC_USERNAME || 'HRL_Agent',
    bridgePort: parseInt(process.env.BRIDGE_PORT) || 8765,
    viewerPort: parseInt(process.env.VIEWER_PORT) || 3007,
    version: process.env.MC_VERSION || '1.20.1'  // Specify version for stability
};

console.log('='.repeat(60));
console.log('Minecraft Hierarchical RL Agent');
console.log('='.repeat(60));
console.log(`Configuration:`);
console.log(`  Server: ${config.host}:${config.port}`);
console.log(`  Username: ${config.username}`);
console.log(`  Bridge Port: ${config.bridgePort}`);
console.log(`  Viewer Port: ${config.viewerPort}`);
console.log(`  MC Version: ${config.version}`);
console.log('='.repeat(60));

// Create the bot
const bot = mineflayer.createBot({
    host: config.host,
    port: config.port,
    username: config.username,
    version: config.version,
    auth: 'offline'  // For local servers without auth
});

// Global references
let skillManager = null;
let bridge = null;

// Bot event handlers — use once() so bridge/viewer only start on first spawn
// (avoids EADDRINUSE crash if the bot dies and respawns during a training run)
bot.once('spawn', () => {
    console.log('[Bot] Spawned in world!');
    console.log(`[Bot] Position: ${bot.entity.position}`);
    console.log(`[Bot] Health: ${bot.health}, Food: ${bot.food}`);

    // Safety check: if spawned in water or void, find dry ground within 64 blocks
    setTimeout(async () => {
        try {
            const pos = bot.entity.position;
            const block = bot.blockAt(pos);
            const blockBelow = bot.blockAt(pos.offset(0, -1, 0));
            const inWater = block?.name?.includes('water') || blockBelow?.name?.includes('water');
            if (inWater || pos.y < 0) {
                console.log(`[Bot] Unsafe spawn (${block?.name} at y=${pos.y.toFixed(1)}), searching for dry ground...`);
                // Search for any non-water, non-air solid block nearby to stand on
                const safeBlock = bot.findBlock({
                    matching: b => b.name === 'grass_block' || b.name === 'dirt' || b.name === 'sand' || b.name === 'stone',
                    maxDistance: 64
                });
                if (safeBlock) {
                    const { pathfinder, Movements, goals } = require('mineflayer-pathfinder');
                    const mcData = require('minecraft-data')(bot.version);
                    const move = new Movements(bot, mcData);
                    move.canSwim = false;
                    bot.pathfinder.setMovements(move);
                    await bot.pathfinder.goto(new goals.GoalNear(safeBlock.position.x, safeBlock.position.y, safeBlock.position.z, 2));
                    console.log(`[Bot] Moved to safe ground at ${bot.entity.position}`);
                } else {
                    console.log('[Bot] Could not find safe ground nearby — bot may be stuck in water');
                }
            } else {
                console.log(`[Bot] Spawn looks safe: ${block?.name} at y=${pos.y.toFixed(1)}`);
            }
        } catch (err) {
            console.error('[Bot] Safe-spawn check failed:', err.message);
        }
    }, 3000); // Wait 3s for chunks to load before checking

    // Initialize skill manager
    skillManager = new SkillManager(bot);
    console.log(`[Bot] Skill Manager initialized with ${skillManager.getActionSpaceSize()} skills`);

    // Start the bridge
    bridge = new Bridge(bot, skillManager, config.bridgePort);
    bridge.start();

    // Initialize prismarine viewer
    try {
        mineflayerViewer(bot, { port: config.viewerPort, firstPerson: true });
        console.log(`[Viewer] Prismarine viewer started at http://localhost:${config.viewerPort}`);
        console.log(`[Viewer] Open this URL in your browser to watch the bot!`);
    } catch (err) {
        console.error('[Viewer] Failed to start prismarine viewer:', err.message);
        console.log('[Viewer] Bot will continue without viewer');
    }

    console.log('[Bot] Ready for Python agent connection!');
});

bot.on('health', () => {
    // Log significant health changes
    if (bot.health <= 5) {
        console.log(`[Bot] WARNING: Low health! (${bot.health})`);
    }
});

bot.on('death', () => {
    console.log('[Bot] Died! Auto-respawning...');
    setTimeout(() => {
        bot.respawn();
    }, 1000);
});

bot.on('kicked', (reason) => {
    console.log(`[Bot] Kicked: ${reason}`);
});

bot.on('error', (err) => {
    console.error('[Bot] Error:', err);
});

bot.on('end', () => {
    console.log('[Bot] Disconnected from server');
    if (bridge) {
        bridge.stop();
    }
    process.exit(0);
});

// Load plugins once bot is injected
bot.once('inject_allowed', () => {
    console.log('[Bot] Loading plugins...');
    bot.loadPlugin(pathfinder);
    bot.loadPlugin(toolPlugin);
    bot.loadPlugin(collectBlock);
    console.log('[Bot] Plugins loaded: pathfinder, tool, collectBlock');
    // Give pathfinder more time to compute paths (default ~5s is too short for
    // large search radii like the 64-block harvest_wood scan)
    bot.pathfinder.thinkTimeout = 15000;
});

// Chat commands for debugging (can be sent from Minecraft)
bot.on('chat', (username, message) => {
    if (username === bot.username) return;
    
    const args = message.split(' ');
    const command = args[0].toLowerCase();
    
    switch (command) {
        case '!skills':
            // List available skills
            const skills = skillManager?.getSkillList() || [];
            bot.chat(`Available skills: ${skills.length}`);
            skills.slice(0, 5).forEach(s => {
                bot.chat(`  ${s.id}: ${s.name} [${s.available ? 'OK' : 'NO'}]`);
            });
            break;
            
        case '!exec':
            // Execute a skill by ID
            const skillId = parseInt(args[1]);
            if (!isNaN(skillId) && skillManager) {
                skillManager.executeSkill(skillId).then(result => {
                    bot.chat(`Result: ${result.success ? 'Success' : 'Failed'} - ${result.message}`);
                });
            }
            break;
            
        case '!state':
            // Print current state summary
            if (bridge) {
                const state = bridge._getState();
                bot.chat(`Pos: ${state.position.x}, ${state.position.y}, ${state.position.z}`);
                bot.chat(`HP: ${state.health}, Food: ${state.food}`);
                bot.chat(`Items: ${Object.keys(state.inventory).length} types`);
            }
            break;
            
        case '!inventory':
            // List inventory
            const inv = bot.inventory.items();
            bot.chat(`Inventory (${inv.length} slots):`);
            inv.slice(0, 5).forEach(item => {
                bot.chat(`  ${item.name}: ${item.count}`);
            });
            break;
            
        case '!help':
            bot.chat('Commands: !skills, !exec <id>, !state, !inventory');
            break;
    }
});

// Graceful shutdown
process.on('SIGINT', () => {
    console.log('\n[Bot] Shutting down...');
    if (bridge) {
        bridge.stop();
    }
    bot.quit();
    process.exit(0);
});

console.log('[Bot] Connecting to server...');
