// /static/api.js
(() => {
  const $ = (id) => document.getElementById(id);
  const API = window.API_BASE || (location.origin.startsWith("http") ? location.origin : "http://srv-ai:7050");

  // ——— Variant A: stateless to the backend ———
  // Отправляем в /chat только system + текущее сообщение пользователя.
  const USE_HISTORY = false;   // НЕ тащим историю в запрос
  const MAX_TURNS   = 8;       // (на будущее) ограничение истории, если включишь USE_HISTORY=true

  const safe = {
    show(el, on) { if (el) el.style.display = on ? "" : "none"; },
    setText(el, txt) { if (el) el.textContent = (txt ?? ""); },
    setHTML(el, html) { if (el) el.innerHTML = (html ?? ""); },
    append(el, node) { if (el && node) el.appendChild(node); },
    clear(el) { if (el) el.innerHTML = ""; },
  };

  // UI refs
  const elChunks  = $("chunks");
  const elSources = $("sources");
  const elThink   = $("think");
  const elModel   = $("model");
  const elStream  = $("stream");
  const elStatus  = $("status");
  const elAPI     = $("api");
  const elErr     = $("err");
  const elChat    = $("chat");
  const elMsg     = $("msg");
  const elSend    = $("send");
  const elClear   = $("clear");

  // чат-история для отображения в UI (бекенду не передаём, см. USE_HISTORY)
  const messages = [
    { role: "system", content:
      "Ты медицинский ассистент. Отвечай ТОЛЬКО на русском языке, без Markdown и кода. " +
      "Не повторяй назначения врача дословно — указывай неточности и дополняй из RAG-контекста."
    }
  ];

  function bubble(role, text) {
    const wrap = document.createElement("div");
    wrap.className = "bubble " + (role === "user" ? "user" : "assistant");
    if (role !== "user") {
      const r = document.createElement("div");
      r.className = "role";
      r.textContent = role === "assistant" ? "Ассистент" : role;
      wrap.appendChild(r);
    }
    const pre = document.createElement("div");
    pre.textContent = text || "";
    wrap.appendChild(pre);
    return { wrap, pre };
  }

  async function fillModels() {
    safe.setText(elAPI, API);
    try {
      const r = await fetch(API + "/runtime/models");
      const m = await r.json();
      safe.clear(elModel);
      const allowed = m.allowed || [];
      const active  = m.active || null;
      allowed.forEach((id) => {
        const opt = document.createElement("option");
        opt.value = id;
        opt.textContent = (m.labels && m.labels[id]) || id;
        if (active && id === active) opt.selected = true;
        safe.append(elModel, opt);
      });
      if (!allowed.length) throw new Error("пустой список моделей");
    } catch {
      ["deepseek-r1:32b","llama3.1:8b"].forEach((id) => {
        const opt = document.createElement("option");
        opt.value = id; opt.textContent = id; safe.append(elModel, opt);
      });
    }
  }

  function scrollChat(){ if (elChat) elChat.scrollTop = elChat.scrollHeight; }
  function extractThink(full){
    if (!full) return "";
    const re = /<think>([\s\S]*?)<\/think>/gi;
    let last=""; let m; while ((m=re.exec(full))!==null) last=m[1];
    return (last||"").trim();
  }
  function setThink(t){ safe.setText(elThink, (t && t.trim()) ? t.trim() : "—"); }

  function setSources(list){
    safe.clear(elSources);
    if (!Array.isArray(list) || list.length===0){
      const li=document.createElement("li"); li.className="muted"; li.textContent="Нет данных";
      elSources && elSources.appendChild(li); return;
    }
    list.slice(0,12).forEach((s)=>{
      const li=document.createElement("li"); li.textContent=String(s);
      elSources && elSources.appendChild(li);
    });
  }

  function setChunks(list){
    if (!elChunks) return;
    safe.clear(elChunks);
    if (!Array.isArray(list) || list.length === 0){
      const li = document.createElement("li");
      li.className = "muted";
      li.textContent = "Нет данных";
      elChunks.appendChild(li);
      return;
    }
    list.slice(0, 8).forEach((c) => {
      const li = document.createElement("li");
      li.style.marginBottom = "10px";
      const head = document.createElement("div");
      head.style.fontWeight = "700";
      head.textContent = `${c.doc_id} • стр. ${c.page_range}`;
      const pre = document.createElement("pre");
      pre.textContent = (c.text || "").trim();
      li.appendChild(head);
      li.appendChild(pre);
      elChunks.appendChild(li);
    });
  }

  // ——— RAG helpers (запрос к превью и цитатам всегда от ТЕКСТА текущего сообщения) ———
  async function fetchChunks(caseText, k = 6){
    try{
      const r = await fetch(API + "/retrieval/preview", {
        method: "POST",
        headers: { "Content-Type": "application/json", "Accept": "application/json" },
        body: JSON.stringify({ case_text: caseText, k, with_text: true })
      });
      if (!r.ok) throw new Error("HTTP " + r.status);
      const js = await r.json();
      const chunks = Array.isArray(js?.chunks) ? js.chunks : [];
      const cits   = Array.isArray(js?.citations) ? js.citations : [];
      setChunks(chunks);
      setSources(cits);
    }catch(e){
      console.warn("fetchChunks failed:", e);
      setChunks([]);
      // источники не трогаем — пусть остаются прежние
    }
  }

  async function fetchCitations(caseText){
    try{
      const r = await fetch(API + "/citations", {
        method:"POST",
        headers:{"Content-Type":"application/json","Accept":"application/json"},
        body: JSON.stringify({ case_text: caseText, k: 6 })
      });
      if (!r.ok) throw new Error("HTTP "+r.status);
      const js = await r.json();
      const c = Array.isArray(js?.citations) ? js.citations : [];
      setSources(c);
    }catch(e){
      console.warn("citations fetch failed:", e);
    }
  }

  async function sendMessage(){
    const text = (elMsg?.value || "").trim();
    if(!text) return;

    elErr.textContent=""; elSend.disabled=true; elStatus.textContent="думает…";

    const {wrap:uWrap} = bubble("user", text); elChat.appendChild(uWrap); scrollChat();
    messages.push({role:"user", content:text});

    const {wrap:aWrap, pre:aPre} = bubble("assistant", ""); elChat.appendChild(aWrap); scrollChat();
    const modelId = elModel?.value || "deepseek-r1:32b";

    // RAG превью/источники — строго от текущего сообщения (stateless UX)
    const caseText = text;
    if (caseText.replace(/\W/g,"").length > 10) {
      fetchChunks(caseText, 6);
    } else {
      setChunks([]);
    }

    try{
      // САМЫЙ ВАЖНЫЙ МОМЕНТ: формируем messages для бэкенда по Variant A
      const payloadMessages = USE_HISTORY
        ? [ messages[0], ...messages.slice(-MAX_TURNS) ]
        : [ messages[0], { role: "user", content: text } ];

      if (elStream && elStream.checked){
        const payload = { model: modelId, messages: payloadMessages, stream:true, temperature:0.4, top_p:0.95 };
        const resp = await fetch(API + "/chat/stream", {
          method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify(payload)
        });
        if (!resp.ok || !resp.body) throw new Error("stream HTTP "+resp.status);

        const reader = resp.body.getReader();
        const decoder = new TextDecoder();
        let buf=""; let full="";

        while(true){
          const {done, value} = await reader.read();
          if (done) break;
          buf += decoder.decode(value, {stream:true});
          let nl;
          while((nl = buf.indexOf("\n")) >= 0){
            const line = buf.slice(0, nl).trim(); buf = buf.slice(nl+1);
            if (!line) continue;
            let evt; try{ evt=JSON.parse(line); }catch{ continue; }
            if (evt.type==="delta"){
              const d=String(evt.delta||""); if(!d) continue;
              full += d;
              // скрываем <think> в выводе
              aPre.textContent = full.replace(/<think>[\s\S]*?<\/think>/gi, "").trim();
              const th = extractThink(full); if (th) setThink(th);
              scrollChat();
            }
          }
        }
        messages.push({role:"assistant", content: aPre.textContent||""});
      } else {
        const payload = { model: modelId, messages: payloadMessages, stream:false, temperature:0.4, top_p:0.95 };
        const r = await fetch(API + "/chat", {
          method:"POST", headers:{"Content-Type":"application/json"}, body: JSON.stringify(payload)
        });
        if (!r.ok) throw new Error("HTTP "+r.status);
        const js = await r.json();
        const msg = String(js?.message || "");
        const vis = msg.replace(/<think>[\s\S]*?<\/think>/gi, "").trim();
        aPre.textContent = vis;
        messages.push({role:"assistant", content: vis});
        const th = extractThink(msg); if (th) setThink(th);
      }

      // Быстрые цитаты отдельно (по текущему сообщению)
      if (caseText.replace(/\W/g,"").length > 10) {
        fetchCitations(caseText);
      }

      elStatus.textContent="готов"; elMsg.value="";
    }catch(e){
      elErr.textContent = "Ошибка: " + (e?.message || e);
      elStatus.textContent="ошибка";
    }finally{
      elSend.disabled=false; scrollChat();
    }
  }

  function clearChat(){
    safe.clear(elChat); setThink("—");
    setSources([]); setChunks([]);
    while(messages.length>1) messages.pop();
    elErr.textContent="";
  }

  if (elSend) elSend.addEventListener("click", sendMessage);
  if (elMsg)  elMsg.addEventListener("keydown", (e)=>{ if(e.key==="Enter" && !e.shiftKey){ e.preventDefault(); sendMessage(); }});
  if (elClear) elClear.addEventListener("click", clearChat);

  document.addEventListener("DOMContentLoaded", ()=>{
    fillModels();
    safe.setText(elAPI, API);
    setThink("—");
  });
})();
