// ============================================
// COLAB KEEP-ALIVE SCRIPT FOR OVERNIGHT TRAINING
// ============================================
// INSTRUCTIONS:
// 1. Mở Google Colab
// 2. Mở Developer Tools (F12 hoặc Ctrl+Shift+I)
// 3. Chuyển sang tab "Console"
// 4. Paste code này và nhấn Enter
// 5. Console sẽ báo "Colab keep-alive script activated!"
// 6. Để Console mở, minimize browser window
// 7. Training sẽ chạy qua đêm mà không bị timeout

function keepColabAlive() {
  console.log('Keeping Colab alive...');
  console.log('Timestamp:', new Date().toISOString());

  // Click connect button if disconnected
  const connectButton = document.querySelector('colab-connect-button');
  if (connectButton && connectButton.shadowRoot) {
    const btn = connectButton.shadowRoot.querySelector('#connect');
    if (btn && btn.getAttribute('is-offline') === 'true') {
      btn.click();
      console.log('✅ Clicked connect button to reconnect');
    }
  }

  // Simulate small activity - scroll notebook slightly
  const notebook = document.querySelector('colab-notebook');
  if (notebook) {
    // Tiny scroll to simulate activity
    window.scrollTo(0, 1);
    setTimeout(() => window.scrollTo(0, 0), 100);
  }
}

// Run keep-alive every 60 minutes (3600000 ms)
const keepAliveInterval = setInterval(keepColabAlive, 60 * 60 * 1000);

console.log('✅ COLAB KEEP-ALIVE SCRIPT ACTIVATED!');
console.log('⏰ Will keep Colab alive every 60 minutes');
console.log('🌙 Safe for overnight training (~7.4 hours)');
console.log('📝 To stop: clearInterval(keepAliveInterval)');

// Initial run
keepColabAlive();

// ============================================
// ADDITIONAL: ACTIVITY SIMULATOR (optional)
// ============================================
// Run this too for extra protection against idle detection

function simulateActivity() {
  // Type into first available cell (without affecting training)
  const cells = document.querySelectorAll('colab-cell-input');
  if (cells.length > 0) {
    const firstCell = cells[0];
    firstCell.focus();
    firstCell.blur();
    console.log('Activity simulated at', new Date().toISOString());
  }
}

// Simulate activity every 30 minutes
const activityInterval = setInterval(simulateActivity, 30 * 60 * 1000);

console.log('✅ ACTIVITY SIMULATOR ACTIVATED!');
console.log('⏰ Will simulate activity every 30 minutes');