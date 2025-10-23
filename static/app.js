// ====== 工具 ======
const qs = s => document.querySelector(s);
const qsa = s => Array.from(document.querySelectorAll(s));

function toast(msg){
  const t = qs('#toast'); if(!t) return;
  t.textContent = msg || '';
  t.classList.add('show');
  setTimeout(()=>t.classList.remove('show'), 1800);
}
function cssPath(el){
  if (!el) return '';
  const path = [];
  while (el && el.nodeType === Node.ELEMENT_NODE){
    let sel = el.nodeName.toLowerCase();
    if (el.id){ sel += `#${el.id}`; path.unshift(sel); break; }
    let sib = el, nth = 1;
    while (sib = sib.previousElementSibling) nth++;
    sel += `:nth-child(${nth})`; path.unshift(sel); el = el.parentElement;
  }
  return path.join('>');
}

// ====== 状态 ======
const STATE = {
  line: (window.__BOOT__ && window.__BOOT__.conf && window.__BOOT__.conf.line) || '松',
  strategy: (window.__BOOT__ && window.__BOOT__.conf && window.__BOOT__.conf.strategy) || 'B',
  mode: (window.__BOOT__ && window.__BOOT__.conf && window.__BOOT__.conf.mode) || 'free',
  session_id: (window.__BOOT__ && window.__BOOT__.session_id) || 's',
  trial_id: (window.__BOOT__ && window.__BOOT__.trial_id) || 't',
  chip: null,
  picker_items: [],
  grid_cols: 3,
  first_sent: false,
  log_buffer: [],
  flush_timer: null,
};

// ====== 日志缓冲 ======
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

// ====== UI ======
function hideHeroOnce(){
  if (STATE.first_sent) return;
  const hero = qs('#hero');
  if (hero){ hero.classList.remove('show'); hero.classList.add('hidden'); }
  STATE.first_sent = true;
}
function autoGrowTextarea(el){
  el.style.height = 'auto';
  const h = Math.min(200, el.scrollHeight);
  el.style.height = h + 'px';
}

function appendUserBubble(text){
  const feed = qs('#feed');
  const row = document.createElement('div');
  row.className = 'msg user fade-in';
  row.innerHTML = `<div class="bubble">${text || '(未输入文本)'}${STATE.chip?`<div style="color:#6B7280;font-size:12px;margin-top:4px">附：${STATE.chip.name}</div>`:''}</div>`;
  feed.appendChild(row);
  feed.scrollTop = feed.scrollHeight;
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
  const span = row.querySelector('.stream');
  span.textContent += delta;
  qs('#feed').scrollTop = qs('#feed').scrollHeight;
}
function finishAssistant(row){
  if (!row) return;
  row.classList.remove('pending');
  const sp = row.querySelector('.spinner'); if(sp) sp.remove();
}

// ====== 线路/模式（事件委托） ======
function bindLines(){
  const panel = qs('.sidebar');
  if (!panel) return;
  panel.addEventListener('click', (e)=>{
    const opt = e.target.closest('.option');
    if(!opt) return;
    qsa('.option').forEach(x=>x.classList.remove('active'));
    opt.classList.add('active');
    const line = opt.dataset.line;
    fetch('/api/set_line', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({line})})
      .then(r=>r.json()).then(j=>{
        STATE.line = j.conf.line; STATE.strategy = j.conf.strategy;
        toast('已切换：' + STATE.line);
        logPush({event:'ui.set_line', detail:{line: STATE.line}});
      })
      .catch(()=> toast('切换失败'));
  });
}
function bindMode(){
  const dd = qs('#fake-model');
  if (!dd) return;
  dd.value = STATE.mode;
  dd.addEventListener('change', ()=>{
    const mode = dd.value;
    fetch('/api/set_mode', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({mode})})
      .then(r=>r.json()).then(j=>{
        STATE.mode = j.conf.mode; STATE.trial_id = j.trial_id;
        toast('已切换任务模式'); logPush({event:'ui.set_mode', detail:{mode: STATE.mode}});
      }).catch(()=> toast('切换失败'));
  });
}

// ====== 选择器 ======
function openPicker(){
  qs('#mask').classList.add('show');
  fetch('/api/picker_list').then(r=>r.json()).then(j=>{
    STATE.picker_items = j.items || []; STATE.grid_cols = j.cols || 3;
    const grid = qs('#picker-grid'); grid.innerHTML='';
    STATE.picker_items.forEach(it=>{
      const cell = document.createElement('div');
      cell.className = 'cell';
      cell.dataset.index = it.index;
      const inner = it.is_image ? `<img src="${it.src}" />`
        : `<div style="height:160px;display:flex;align-items:center;justify-content:center">📄</div>`;
      cell.innerHTML = `${inner}<div class="meta"><div><div class="title">${it.name}</div><div class="size">${it.size}</div></div><div>双击上传</div></div>`;
      cell.ondblclick = ()=> selectCandidate(it.index);
      grid.appendChild(cell);
    });
    logPush({event:'ui.picker_open'});
  });
}
function closePicker(){ qs('#mask').classList.remove('show'); logPush({event:'ui.picker_close'}); }
function selectCandidate(index){
  fetch('/api/pick', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({index})})
    .then(r=>r.json()).then(j=>{
      if(!j.ok) return;
      const d = j.display; STATE.chip = d;
      const chip = qs('#chip'); chip.classList.add('show');
      chip.querySelector('.name').textContent = d.name;
      chip.querySelector('.size').textContent = d.size || '';
      closePicker();
      logPush({event:'ui.pick_display', detail:{index: d.index, name: d.name}});
    });
}

// ====== 发送（流式优先，失败回退非流式） ======
function send(){
  const input = qs('#input');
  const txt = input.value.trim();
  if (!txt && !STATE.chip){ return; }

  hideHeroOnce();
  appendUserBubble(txt);
  logPush({event:'qa.user_send', detail:{text: txt}});

  input.value = ''; autoGrowTextarea(input);

  const row = createAssistantPlaceholder();

  // 超时兜底（防止 done 丢失）
  const safety = setTimeout(()=> { try{ finishAssistant(row); }catch(_){} }, 90000);

  // 首选：流式
  fetch('/api/send_stream', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({text: txt})
  }).then(async (res)=>{
    if (!res.ok || !res.body){
      throw new Error('stream not available');
    }
    const reader = res.body.getReader();
    const decoder = new TextDecoder('utf-8');
    let buffer = '';
    let finalText = '';
    let curEvent = 'delta';

    const handleLine = (line)=>{
      if (!line.trim()) return;
      if (line.startsWith('event:')){
        curEvent = line.slice(6).trim() || 'delta';
        return;
      }
      if (!line.startsWith('data:')) return;

      const raw = line.slice(5).trim(); // 一整行 JSON
      let data; try{ data = JSON.parse(raw); }catch{ data = raw; }

      if (curEvent === 'delta'){
        const txt = typeof data === 'string' ? data : (data && data.t) || '';
        if (txt){ updateAssistantStream(row, txt); finalText += txt; }

      } else if (curEvent === 'meta'){
        if (data && data.toast) toast(data.toast);

      } else if (curEvent === 'modal'){
        if (data) openClarify(data);

      } else if (curEvent === 'done'){
        finishAssistant(row);
        clearTimeout(safety);
        logPush({event:'qa.model_reply', detail:{text: finalText}});
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
    if (buffer){ handleLine(buffer); finishAssistant(row); clearTimeout(safety); }
  }).catch(()=>{
    // 回退：非流式
    fetch('/api/send', {
      method:'POST', headers:{'Content-Type':'application/json'},
      body: JSON.stringify({text: txt})
    }).then(r=>r.json()).then(j=>{
      if (j.toast) toast(j.toast);
      if (j.modal){
        openClarify(j.modal);
        finishAssistant(row); clearTimeout(safety);
        return;
      }
      const plain = j.assistant_html ? j.assistant_html.replace(/<[^>]+>/g,'') : '(空响应)';
      updateAssistantStream(row, plain);
      finishAssistant(row); clearTimeout(safety);
      logPush({event:'qa.model_reply', detail:{html: j.assistant_html || ''}});
    }).catch(()=>{
      updateAssistantStream(row, '(发送失败)');
      finishAssistant(row); clearTimeout(safety);
    });
  });
}

function openClarify(modal){
  const m = qs('#mask'); m.classList.add('show');
  const grid = qs('#picker-grid'); grid.innerHTML='';
  const head = document.createElement('div');
  head.className = 'cell';
  head.innerHTML = `<div class="meta"><div class="title">${modal.title || '请选择'}</div><div class="size">点击确认</div></div>`;
  grid.appendChild(head);
  (modal.options || []).forEach(opt=>{
    const d = document.createElement('div');
    d.className = 'cell';
    d.innerHTML = `<div class="meta"><div class="title">${opt}</div><div class="size">选择</div></div>`;
    d.onclick = ()=>{ closePicker(); toast('已选择：'+opt); logPush({event:'ui.clarify_choose', detail:{option: opt}}); };
    grid.appendChild(d);
  });
}

// ====== 全量行为采集（不阻塞） ======
function bindCapture(){
  document.addEventListener('click', (e)=>{ logPush({event:'dom.click', detail:{x:e.clientX,y:e.clientY,button:e.button,path:cssPath(e.target)}}); }, true);
  document.addEventListener('dblclick', (e)=>{ logPush({event:'dom.dblclick', detail:{x:e.clientX,y:e.clientY,path:cssPath(e.target)}}); }, true);
  document.addEventListener('keydown', (e)=>{ logPush({event:'kbd.keydown', detail:{key:e.key,code:e.code,alt:e.altKey,ctrl:e.ctrlKey,shift:e.shiftKey}}); }, true);
  document.addEventListener('keyup',   (e)=>{ logPush({event:'kbd.keyup',   detail:{key:e.key,code:e.code}}); }, true);
  document.addEventListener('input', (e)=>{
    if (!(e.target instanceof HTMLInputElement) && !(e.target instanceof HTMLTextAreaElement)) return;
    const t = e.target;
    logPush({event:'form.input', detail:{path:cssPath(t), value:t.value, selStart:t.selectionStart, selEnd:t.selectionEnd}});
  }, true);
  document.addEventListener('change', (e)=>{ logPush({event:'form.change', detail:{path:cssPath(e.target), value:e.target.value}}); }, true);
}

// ====== 初始化 ======
document.addEventListener('DOMContentLoaded', ()=>{
  try{
    bindLines();
    bindMode();
    bindCapture();

    const input = qs('#input');
    autoGrowTextarea(input);
    input.addEventListener('input', ()=> autoGrowTextarea(input));
    input.addEventListener('keydown', (e)=>{
      if (e.key === 'Enter' && !e.shiftKey && !e.isComposing){
        e.preventDefault();
        send();
      }
    });

    qs('#send').addEventListener('click', ()=> send());
    qs('#paperclip').addEventListener('click', ()=> openPicker());
    qs('#close').addEventListener('click', ()=> closePicker());

    logPush({event:'boot', detail:{session_id: STATE.session_id, trial_id: STATE.trial_id, mode: STATE.mode, line: STATE.line}});
  }catch(err){
    console.error(err);
    toast('前端脚本初始化失败');
  }
});
