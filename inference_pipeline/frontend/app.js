const API_BASE_URL = (window.__API_BASE__ || "http://localhost:8000").replace(/\/+$/, "");
const LANG_PLACEHOLDERS = {
  en: "Ask your question...",
  ta: "உங்கள் கேள்வியை கேளுங்கள்...",
  hi: "अपना प्रश्न पूछें...",
  bn: "আপনার প্রশ্ন জিজ্ঞাসা করুন...",
  te: "మీ ప్రశ్న అడగండి...",
  ml: "നിങ്ങളുടെ ചോദ്യമുയര്‍ത്തുക...",
  kn: "ನಿಮ್ಮ ಪ್ರಶ್ನೆಯನ್ನು ಕೇಳಿ...",
  mr: "आपला प्रश्न विचारा...",
  gu: "તમારો પ્રશ્ન પૂછો..."
};
const MSG = {
  en: { no_info: "No authoritative information found.", ask: "Ask", wait: "Searching..." },
  ta: { no_info: "அதிகாரப்பூர்வ தகவல் இல்லை.", ask: "கேளுங்கள்", wait: "தேடுகிறது..." },
  hi: { no_info: "प्राधिकृत जानकारी नहीं मिली.", ask: "पूछें", wait: "खोज रहा है..." }
};
let sessionId = crypto.randomUUID ? crypto.randomUUID() : `${Date.now()}-${Math.random()}`;
let history = [];
function $(id){ return document.getElementById(id); }
function escapeHtml(s){ return String(s).replace(/[&<>"']/g, c => ({ '&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;' })[c]); }
function setStatus(text, isError=false, busy=false){ const st = $("status"); st.style.color = isError ? "#b00020" : "#444"; st.innerHTML = busy ? `<span class="spinner"></span>${escapeHtml(text||"")}` : escapeHtml(text||""); }
function enableUi(enable){ $("askBtn").disabled = !enable; $("query").disabled = !enable; $("lang").disabled = !enable; }
function applyPlaceholder(){ const lang = $("lang").value || "en"; $("query").placeholder = LANG_PLACEHOLDERS[lang] || LANG_PLACEHOLDERS.en; $("askBtn").textContent = (MSG[lang] && MSG[lang].ask) || "Ask"; }
function shortDomain(url){ try{ return new URL(url).hostname.replace(/^www\./,""); }catch{ return url||""; } }
function renderAnswerLines(lines){ const out = $("answers"); out.innerHTML = ""; (lines||[]).forEach((ln,i)=>{ const div=document.createElement("div"); div.className="answer-line"; const num=document.createElement("strong"); num.textContent=`${i+1}. `; const span=document.createElement("span"); span.textContent = (typeof ln==="string")?ln:(ln.text||""); div.appendChild(num); div.appendChild(span); out.appendChild(div); }); $("responseCard").hidden = false; $("responseCard").focus(); }
function renderSources(citations){ const out=$("sources"); out.innerHTML=""; const seen=new Set(); (citations||[]).forEach(c=>{ const url=c.source_url; if(!url||seen.has(url)) return; seen.add(url); const div=document.createElement("div"); div.innerHTML=`<span>[${c.citation}]</span> <a class="source-link" target="_blank" href="${escapeHtml(url)}">${escapeHtml(shortDomain(url))}</a>`; out.appendChild(div); }); }
function pushHistory(q){ history.unshift({ ts: Date.now(), q }); $("history").innerHTML=""; history.slice(0,40).forEach(item=>{ const el=document.createElement("div"); el.textContent = `${new Date(item.ts).toLocaleString()} • ${item.q}`; $("history").appendChild(el); }); }
async function ask(){ const q=$("query").value.trim(); if(!q){ setStatus("Please enter a question.", true); return; } const lang=$("lang").value||"en"; const reqId = crypto.randomUUID ? crypto.randomUUID() : `${Date.now()}`; const payload = { channel: "web", language: lang, query: q, session_id: sessionId, request_id: reqId }; enableUi(false); setStatus((MSG[lang] && MSG[lang].wait) || "Searching...", false, true); try{ const controller = new AbortController(); const timeout = setTimeout(()=>controller.abort(),15000); const resp = await fetch(`${API_BASE_URL}/v1/query`,{ method:"POST", headers:{"Content-Type":"application/json; charset=utf-8"}, body:JSON.stringify(payload), signal:controller.signal }); clearTimeout(timeout); if(!resp.ok){ const txt = await resp.text().catch(()=> ""); setStatus(`Server error ${resp.status} ${txt}`, true); enableUi(true); return; } const body = await resp.json(); if(body && body.resolution === "answer"){ const lines = (body.answer_lines||[]).map(l=> typeof l==="string"?{text:l}:l); renderAnswerLines(lines); renderSources(body.citations||[]); setStatus("Answer ready"); pushHistory(q); enableUi(true); return; } setStatus((MSG[lang] && MSG[lang].no_info) || "No authoritative information found.", true); $("answers").innerHTML = ""; $("sources").innerHTML = ""; pushHistory(q); }catch(err){ setStatus(err && err.name === "AbortError" ? "Request timed out." : "Network error.", true); }finally{ enableUi(true); } }
document.addEventListener("DOMContentLoaded", ()=>{ $("askBtn").addEventListener("click", ask); $("query").addEventListener("keydown", e=>{ if(e.key==="Enter" && (e.ctrlKey||e.metaKey)) ask(); }); $("lang").addEventListener("change", applyPlaceholder); $("clearHistory").addEventListener("click", ()=>{ history=[]; $("history").innerHTML=""; setStatus("History cleared"); }); applyPlaceholder(); setStatus(""); });
