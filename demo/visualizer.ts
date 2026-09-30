import { BlockAllocator } from '../src/core/block_allocator';
import { SequenceManager } from '../src/core/sequence_manager';

const allocator = new BlockAllocator(64, 4);
const manager = new SequenceManager(allocator);

const gridEl = document.getElementById('block-grid');
const statsEl = document.getElementById('pool-stats');
const infoEl = document.getElementById('seq-info');

function render() {
  if (!gridEl || !statsEl || !infoEl) return;

  statsEl.textContent = `Allocated: ${allocator.getAllocatedCount()} / ${allocator.totalBlocks} blocks (Free: ${allocator.getFreeBlockCount()})`;

  gridEl.innerHTML = '';
  for (let i = 0; i < allocator.totalBlocks; i++) {
    const el = document.createElement('div');
    el.className = 'block';
    const refCount = allocator.getRefCount(i);

    if (refCount === 0) {
      el.classList.add('free');
      el.textContent = `${i}`;
    } else if (refCount === 1) {
      el.classList.add('inuse');
      el.textContent = `${i}`;
    } else {
      el.classList.add('shared');
      el.textContent = `${i} (x${refCount})`;
    }
    gridEl.appendChild(el);
  }

  infoEl.innerHTML = `Active Sequences: ${manager.getActiveCount()}`;
}

document.getElementById('btn-add-seq')?.addEventListener('click', () => {
  try {
    const promptLen = 5 + Math.floor(Math.random() * 8);
    const tokens = Array.from({ length: promptLen }, (_, i) => i + 1);
    manager.createSequence(tokens);
    render();
  } catch (err: any) {
    alert(err.message);
  }
});

let activeSeqIds: number[] = [];

document.getElementById('btn-step-token')?.addEventListener('click', () => {
  for (const id of activeSeqIds) {
    try {
      manager.appendToken(id, Math.floor(Math.random() * 1000));
    } catch {}
  }
  render();
});

render();
