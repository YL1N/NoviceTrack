/* NoviceTrack 前端交互
 * - 选择器弹窗：固定尺寸 + 5列 + 懒加载 + 占位 + 多选 + 删除
 * - 输入区附件 chip：多项、可删
 * - 发送：Enter（Shift+Enter 换行），流式 + 暂停/中断（AbortController）
 * - 只发附件：自动生成默认说明；用户气泡上方显示预览图/文件卡片
 * - 对话区滚动容器 #feed
 */

const qs = s => document.querySelector(s);
const qsa = s => Array.from(document.querySelectorAll(s));

function toast(msg){
  const t = qs('#toast');
  if(!t) return;
  t.textContent = msg || '';
  t.classList.add('show');
  setTimeout(()=>t.classList.remove('show'), 1500);
}

function cssPath(el){
  if (!el) return '';
  const path = [];
  while (el && el.nodeType === Node.ELEMENT_NODE){
    let sel = el.nodeName.toLowerCase();
    if (el.id){
      sel += `#${el.id}`;
      path.unshift(sel);
      break;
    }
    let sib = el, nth = 1;
    while (sib = sib.previousElementSibling) nth++;
    sel += `:nth-child(${nth})`;
    path.unshift(sel);
    el = el.parentElement;
  }
  return path.join('>');
}

const STATE = {
line: (window.__BOOT__ && window.__BOOT__.conf && window.__BOOT__.conf.line) || '松',
strategy: (window.__BOOT__ && window.__BOOT__.conf && window.__BOOT__.conf.strategy) || 'B',
mode: (window.__BOOT__ && window.__BOOT__.conf && window.__BOOT__.conf.mode) || 'free',
session_id: (window.__BOOT__ && window.__BOOT__.session_id) || 's',
trial_id: (window.__BOOT__ && window.__BOOT__.trial_id) || 't',

  picker_items: [],
  chips: [], // {index,name,size,src,is_image}
  streaming: false,
  controller: null,
  pendingRow: null,
  log_buffer: [],
  flush_timer: null,
  first_sent: false,
};

function logPush(ev){
  try{
    STATE.log_buffer.push(ev);
    if (!STATE.flush_timer){
      STATE.flush_timer = setTimeout(()=>{
        const events = STATE.log_buffer.splice(0, STATE.log_buffer.length);
        STATE.flush_timer = null;
        fetch('/api/log', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({events})});
      }, 700);
    }
  }catch(_){}
}

function hideHeroOnce(){
  if (STATE.first_sent) return;
  (function(el){ if(el) el.classList.add('hidden'); })(qs('#hero'));
  STATE.first_sent = true;
}

function autoGrowTextarea(el){
  el.style.height = 'auto';
  el.style.height = Math.min(200, el.scrollHeight) + 'px';
}

/* ========== 附件 chip 渲染（修复：使用现有 #chip，避免 #chips null 报错） ========== */
/* ========== 附件 chip 渲染（多项 + 可删除，实际取消后端 picks） ========== */
function renderChips(){
  const wrap = qs('#chip');
  const list = qs('#chip-list');
  if(!wrap || !list) return;

  list.innerHTML = '';

  if (STATE.chips.length === 0){
    wrap.classList.remove('show');
    return;
  }

  wrap.classList.add('show');

  STATE.chips.forEach((c, i)=>{
    const token = document.createElement('div');
    token.className = 'chip-token';

    const thumb = c.is_image
      ? `<div class="thumb-sm"><img src="${c.src}" onerror="this.style.opacity=.3;"></div>`
      : `<div class="thumb-sm">📄</div>`;

    token.innerHTML = `
      ${thumb}
      <div class="meta-sm">
        <div class="name" title="${c.name}">${c.name}</div>
        <div class="size">${c.size || ''}</div>
      </div>
      <button class="x" title="取消这项">×</button>
    `;

    // 单项删除：前端移除 + 后端真正移除（避免被带着发过去）
    token.querySelector('.x').onclick = ()=>{
      // 前端列表去掉
      STATE.chips.splice(i, 1);
      renderChips();
      // 通知后端移除该 pick（以 display.index 为键）
      fetch('/api/remove_pick', {
        method:'POST',
        headers:{'Content-Type':'application/json'},
        body: JSON.stringify({index: c.index})
      }).catch(()=>{ /* 忽略网络抖动；用户 UI 已更新 */ });
    };

    list.appendChild(token);
  });
}


/* 选择器中标记已加入（小角标），只影响弹窗网格 */
function markCellAdded(idx){
  const el = qs(`.grid .cell[data-index="${idx}"]`);
  if (el) el.classList.add('added');
}

/* ========== 对话气泡 ========== */
function renderAttachPreviewHTML(items){
  if (!items || !items.length) return '';
  const cells = items.map(it=>{
    if (it.is_image){
      const src = it.b64 || it.src;
      if (src){
        return `<div class="pv"><img src="${src}" loading="eager" onerror="this.closest('.pv').innerHTML='<div class=&quot;pv file&quot;><div class=&quot;icon&quot;>📄</div><div class=&quot;fn&quot;>${(it.name||'图片')}</div></div>';"></div>`;
      }
      return `<div class="pv file"><div class="icon">📄</div><div class="fn" title="${it.name}">${it.name}</div></div>`;
    }
    return `<div class="pv file"><div class="icon">📄</div><div class="fn" title="${it.name}">${it.name}</div></div>`;
  }).join('');
  return `<div class="preview-bar">${cells}</div>`;
}


/* 修复：预览叠在气泡上方（不再挤到左侧抬高气泡） */
function appendUserBubble(text, attaches){
  const feed = qs('#feed');
  const row = document.createElement('div');
  row.className = 'msg user fade-in';
  const pv = renderAttachPreviewHTML(attaches);
  row.innerHTML = `<div class="stack">${pv}<div class="bubble">${text || '(未输入文本)'}</div></div>`;
  feed.appendChild(row);
  feed.scrollTop = feed.scrollHeight;
  return row;
}


function injectPreviews(userRow, previews){
  if (!userRow || !previews || !previews.length) return;
  const stack = userRow.querySelector('.stack');
  if (!stack) return;

  // 生成预览 HTML（优先 b64，再退回 src；失败时给占位卡）
  const cells = previews.map(p=>{
    if (p.is_image){
      const src = p.b64 || p.src;
      if (src){
        return `<div class="pv"><img src="${src}" loading="eager" onerror="this.closest('.pv').innerHTML='<div class=&quot;pv file&quot;><div class=&quot;icon&quot;>📄</div><div class=&quot;fn&quot;>${(p.name||'图片')}</div></div>';"></div>`;
      }
      return `<div class="pv file"><div class="icon">📄</div><div class="fn">${(p.name||'图片')}</div></div>`;
    }
    return `<div class="pv file"><div class="icon">📄</div><div class="fn">${(p.name||'文件')}</div></div>`;
  }).join('');
  const bar = document.createElement('div');
  bar.className = 'preview-bar';
  bar.innerHTML = cells;

  // 如果已有预览条，先移除再插入新条
  const old = stack.querySelector('.preview-bar');
  if (old) old.remove();
  stack.insertBefore(bar, stack.firstChild);
}


function createAssistantPlaceholder(){
  const feed = qs('#feed');
  const row = document.createElement('div');
  row.className = 'msg assistant pending fade-in';
  row.innerHTML = `<div class="spinner"></div><div class="bubble"><span class="stream"></span></div>`;
  feed.appendChild(row);
  feed.scrollTop = feed.scrollHeight;
  return row;
}

function updateAssistantStream(row, delta){
  row.querySelector('.stream').textContent += delta;
  qs('#feed').scrollTop = qs('#feed').scrollHeight;
}

function finishAssistant(row){
  if (!row) return;
  row.classList.remove('pending');
  (function(sp){ if(sp){ sp.remove(); } })(row.querySelector('.spinner'));
}

/* ========== 发送按钮态：发送 ↔ 暂停/中断 ========== */
function setSendButtonStreaming(b){
  const btn = qs('#send');
  STATE.streaming = b;
  if (b){
    btn.classList.add('pause');
    btn.setAttribute('title','中断当前回答');
    btn.innerHTML = '⏸';
  }else{
    btn.classList.remove('pause');
    btn.setAttribute('title','发送');
    btn.innerHTML = '➤';
  }
}

/* ========== 线路/模式 ========== */
function bindLines(){
  qs('.sidebar')?.addEventListener('click', (e)=>{
    const opt = e.target.closest('.option');
    if(!opt) return;
    qsa('.option').forEach(x=>x.classList.remove('active'));
    opt.classList.add('active');
    const line = opt.dataset.line;
    fetch('/api/set_line', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({line})})
      .then(r=>r.json()).then(j=>{
        STATE.line = j.conf.line;
        STATE.strategy = j.conf.strategy;
        toast('已切换：'+STATE.line);
      });
  });
}

function bindMode(){
  const dd = qs('#fake-model');
  if (!dd) return;
  dd.value = STATE.mode;

  dd.addEventListener('change', ()=>{
    const mode = dd.value;

    fetch('/api/set_mode', {
      method:'POST',
      headers:{'Content-Type':'application/json'},
      body: JSON.stringify({mode})
    })
    .then(r=>r.json())
    .then(j=>{
      STATE.mode = j.conf.mode;
      STATE.trial_id = j.trial_id;

      // 1) 前端清空待发附件（很关键，否则不会重新触发 /api/pick）
      STATE.chips = [];
      renderChips();

      // 2) 后端也清空 picks，免得会话里残留老选择
      fetch('/api/clear_picks', {
        method:'POST',
        headers:{'Content-Type':'application/json'}
      });

      // 3) 轻提示
      const label = mode==='task_i' ? '任务I'
                  : mode==='task_ii' ? '任务II'
                  : mode==='task_iii' ? '任务III'
                  : '自由模式';
      toast('已切换：' + label);
    });
  });
}


/* ========== 选择器（固定大小 + 5列 + 懒加载 + 多选） ========== */
function openPicker(){
  qs('#mask').classList.add('show');
  fetch('/api/picker_list').then(r=>r.json()).then(j=>{
    STATE.picker_items = j.items || [];
    const grid = qs('#picker-grid');
    grid.innerHTML='';

    // 打开前清理旧的“已选择”视觉状态
    // （避免新对话后仍然显示已选择）
    // 这里直接重建 DOM 已经会清，但保险起见再清一次：
    // （如果外部自定义样式有残留）
    // 无需处理

    STATE.picker_items.forEach(it=>{
      const cell = document.createElement('div');
      cell.className = 'cell';
      cell.dataset.index = it.index;
      cell.dataset.rel = it.rel || '';

      const thumbSrc = it.is_image
        ? `/thumb/${encodeURIComponent(it.rel || '')}?w=360`
        : null;

      const imgHTML = it.is_image
        ? `<img class="thumb" loading="lazy" src="${thumbSrc}"
             alt="${it.name}"
             onerror="this.onerror=null;this.closest('.thumb-wrap').innerHTML='<div class=&quot;file-icon&quot;>📄</div>';">`
        : `<div class="file-icon">📄</div>`;

      cell.innerHTML = `
        <div class="thumb-wrap">${imgHTML}<div class="badge">双击上传</div></div>
        <div class="meta">
          <div class="title" title="${it.name}">${it.name}</div>
          <div class="size">${it.size||''}</div>
        </div>`;

      cell.ondblclick = ()=> selectCandidate(it);
      grid.appendChild(cell);
    });
  });
}


function closePicker(){
  qs('#mask').classList.remove('show');
}

/* 修复点：双击后
   1) 立刻关闭弹窗
   2) 前端去重；即使后端判重也会在 UI 显示
   3) 更新 #chip 以给用户明确反馈
*/
function selectCandidate(it){
  fetch('/api/pick', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({index: it.index})
  })
  .then(r=>r.json())
  .then(j=>{
    if(!j || j.ok === false) return;

    // UI 上始终以“用户手动点的那张”为准做高亮
    markCellAdded(it.index);

    // 构建待发 CHIP（仍然推入服务器最终选择对象的展示信息；若服务器返回 actual，就用它的 name/size/src）
    const chosen = (j.actual && STATE.mode === 'task_i') ? j.actual : it;

    // 去重（以展示/发送列表为准）
    if (STATE.chips.some(x => x.index === chosen.index)){
      toast('已在待发送列表');
      closePicker();
      return;
    }

    STATE.chips.push({
      index: chosen.index,
      name: chosen.name,
      size: chosen.size,
      is_image: chosen.is_image,
      // 预览优先用缩略图；后备用原 src
      src: chosen.is_image ? `/thumb/${encodeURIComponent(chosen.rel || chosen.name)}?w=360` : (chosen.src || '')
    });

    renderChips();
    toast(j.dup ? '已在待发送列表' : '已加入待发送');
    closePicker();
  });
}



/* ========== 发送（流式优先） ========== */
function send(){
  // 正在流式 → 这次点击当作“中断”
  if (STATE.streaming && STATE.controller){
    try{ STATE.controller.abort(); }catch(_){}
    setSendButtonStreaming(false);
    finishAssistant(STATE.pendingRow);
    STATE.pendingRow = null;
    toast('已中断当前回答');
    return;
  }

  const input = qs('#input');
  let txt = input.value.trim();
  if (!txt && STATE.chips.length===0) return;

  // 仅附件：默认文案
  if (!txt && STATE.chips.length>0){
    txt = "请基于我刚刚附带的文件或图片，进行有用的解读、摘要与建议；如需明确目标，请先用一句话澄清后再回答。";
  }

  hideHeroOnce();

  // 渲染用户气泡（含本地快照预览）
  const usedChips = STATE.chips.slice();
  const rowUser = appendUserBubble(txt, usedChips);
  STATE.lastUserRow = rowUser;

  // 清空输入与本地附件 UI（后端仍保留 picks）
  input.value = '';
  autoGrowTextarea(input);
  STATE.chips = [];
  renderChips();

  const row = createAssistantPlaceholder();
  STATE.pendingRow = row;

  // 启动流式（只创建一次）
  STATE.controller = new AbortController();
  setSendButtonStreaming(true);

  // —— 首包看门狗（3.5s 未拿到任何数据就兜底）
  let started = false;
  const startFallback = () => {
    if (started) return;
    try { STATE.controller.abort(); } catch (_) {}
    fetch('/api/send', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({text: txt})
    }).then(r=>r.json()).then(j=>{
      const text2 = j.assistant_text || '（空响应）';
      updateAssistantStream(row, text2);
      finishAssistant(row);
      setSendButtonStreaming(false);
      STATE.pendingRow = null;
    }).catch(()=>{
      updateAssistantStream(row, '（发送失败）');
      finishAssistant(row);
      setSendButtonStreaming(false);
      STATE.pendingRow = null;
    });
  };
  const SSE_FIRST_CHUNK_TIMEOUT_MS = 3500;
  const preTimer = setTimeout(startFallback, SSE_FIRST_CHUNK_TIMEOUT_MS);

  fetch('/api/send_stream', {
    method:'POST',
    headers:{'Content-Type':'application/json'},
    body: JSON.stringify({text: txt}),
    signal: STATE.controller.signal
  }).then(async (res)=>{
    started = true;
    clearTimeout(preTimer);

    if (!res.ok || !res.body){
      throw new Error('stream not available');
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buffer = '';
    let curEvent = 'delta';
    let gotDelta = false;

    const handleLine = (line)=>{
      if (!line.trim()) return;
      if (line.startsWith('event:')){
        curEvent = line.slice(6).trim() || 'delta';
        return;
      }
      if (!line.startsWith('data:')) return;
      const raw = line.slice(5).trim();
      let data;
      try{ data = JSON.parse(raw); }catch{ data = raw; }

      if (curEvent === 'delta'){
        const d = typeof data === 'string' ? data : (data && data.t) || '';
        if (d){
          gotDelta = true;
          updateAssistantStream(row, d);
        }
      } else if (curEvent === 'meta'){
        if (data && data.toast) toast(data.toast);
        if (data && data.previews && STATE.lastUserRow){
          injectPreviews(STATE.lastUserRow, data.previews);  // ★ 服务端首包回填预览
        }
      } else if (curEvent === 'modal'){
        toast((data && data.title) || '需要澄清');
      } else if (curEvent === 'done'){
        // ignore
      }
    };

    while(true){
      const {done, value} = await reader.read();
      if (done) break;
      const chunk = decoder.decode(value, {stream:true});
      const parts = (buffer + chunk).split('\n');
      buffer = parts.pop();
      for (const ln of parts){ handleLine(ln); }
    }
    if (buffer){ handleLine(buffer); }

    finishAssistant(row);
    setSendButtonStreaming(false);
    STATE.pendingRow = null;

    if (!gotDelta) {
      try{
        const j = await fetch('/api/send', {
          method:'POST',
          headers:{'Content-Type':'application/json'},
          body: JSON.stringify({text: txt})
        }).then(r=>r.json());
        if (j.toast) toast(j.toast);
        const text2 = j.assistant_text || '（空响应）';
        updateAssistantStream(row, text2);
      }catch(_){
        updateAssistantStream(row, '（发送失败）');
      }
    }
  }).catch((err)=>{
    if (err.name === 'AbortError') return;
    startFallback();
  });
}



function startNewChat(){
  // 1) 如果正在流式，先中断
  if (STATE.streaming && STATE.controller){
    try{ STATE.controller.abort(); }catch(_){}
    setSendButtonStreaming(false);
    finishAssistant(STATE.pendingRow);
    STATE.pendingRow = null;
  }

  // 2) 清空输入/附件/本地状态
  const input = qs('#input');
  input.value = '';
  autoGrowTextarea(input);

  STATE.chips = [];
  renderChips();

  STATE.first_sent = false;

  // 3) 清空对话区并回到首屏
  const feed = qs('#feed');
  if (feed) feed.innerHTML = '';
  const hero = qs('#hero');
  if (hero){
    hero.classList.remove('hidden');
    hero.classList.add('show');
  }

  // 4) 关闭选择器并清空网格，移除任何“已选择”标记
  qs('#mask').classList.remove('show');
  const grid = qs('#picker-grid');
  if (grid){ grid.innerHTML = ''; }
  // 防守式：如果外界仍保留了旧 DOM
  qsa('.grid .cell.added').forEach(el=>el.classList.remove('added'));

  // 5) 通知后端重置服务端会话
  fetch('/api/new_chat', {method:'POST'})
    .then(r => r.json())
    .then(j => {
      if (j && j.ok){
        toast('已开始新对话');
      }else{
        toast('新对话初始化失败');
      }
    })
    .catch(()=> toast('新对话初始化失败'));
}


/* ========== 捕获行为日志（点击/键盘/输入等） ========== */
function bindCapture(){
  document.addEventListener('click', (e)=>{
    logPush({event:'dom.click', detail:{x:e.clientX,y:e.clientY,button:e.button,path:cssPath(e.target)}});
  }, true);
  document.addEventListener('dblclick', (e)=>{
    logPush({event:'dom.dblclick', detail:{x:e.clientX,y:e.clientY,path:cssPath(e.target)}});
  }, true);
  document.addEventListener('keydown', (e)=>{
    logPush({event:'kbd.keydown', detail:{key:e.key,code:e.code,alt:e.altKey,ctrl:e.ctrlKey,shift:e.shiftKey}});
  }, true);
  document.addEventListener('keyup', (e)=>{
    logPush({event:'kbd.keyup', detail:{key:e.key,code:e.code}});
  }, true);
  document.addEventListener('input', (e)=>{
    if (!(e.target instanceof HTMLInputElement) && !(e.target instanceof HTMLTextAreaElement)) return;
    const t = e.target;
    logPush({event:'form.input', detail:{path:cssPath(t), value:t.value, selStart:t.selectionStart, selEnd:t.selectionEnd}});
  }, true);
  document.addEventListener('change', (e)=>{
    logPush({event:'form.change', detail:{path:cssPath(e.target), value:e.target.value}});
  }, true);
}

/* ========== 初始化 ========== */
document.addEventListener('DOMContentLoaded', ()=>{
  bindLines();
  bindMode();
  bindCapture();

  const input = qs('#input');
  autoGrowTextarea(input);
  input.addEventListener('input', ()=> autoGrowTextarea(input));
  input.addEventListener('keydown', (e)=>{
    // Enter 发送（Shift+Enter 换行）; 流式中按 Enter = 中断
    if (e.key === 'Enter' && !e.shiftKey && !e.isComposing){
      e.preventDefault();
      send();
    }
  });

  qs('#send').addEventListener('click', ()=> send());
  qs('#paperclip').addEventListener('click', ()=> openPicker());
  qs('#close').addEventListener('click', ()=> closePicker());
  qs('#btn-current')?.addEventListener('click', startNewChat); // ← 绑定“新对话”按钮
});

