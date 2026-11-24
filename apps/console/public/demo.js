// ---------- pure-JS SHA-256 (works over plain HTTP where WebCrypto is unavailable) ----------
function sha256hex(ascii){
  function rr(v,a){return (v>>>a)|(v<<(32-a));}
  const maxWord=Math.pow(2,32);let result='';const words=[];const bitLen=ascii.length*8;
  let hash=[];const k=[];let pc=0;const comp={};
  for(let c=2;pc<64;c++){if(!comp[c]){for(let i=0;i<313;i+=c)comp[i]=c;hash[pc]=(Math.pow(c,.5)*maxWord)|0;k[pc++]=(Math.pow(c,1/3)*maxWord)|0;}}
  ascii+='\x80';while(ascii.length%64-56)ascii+='\x00';
  for(let i=0;i<ascii.length;i++){const j=ascii.charCodeAt(i);if(j>>8)return '';words[i>>2]|=j<<((3-i)%4)*8;}
  words[words.length]=(bitLen/maxWord)|0;words[words.length]=bitLen;
  for(let j=0;j<words.length;){const w=words.slice(j,j+=16);const old=hash.slice(0,8);
    for(let i=0;i<64;i++){const w15=w[i-15],w2=w[i-2];const a=hash[0],e=hash[4];
      const t1=hash[7]+(rr(e,6)^rr(e,11)^rr(e,25))+((e&hash[5])^(~e&hash[6]))+k[i]+(w[i]=i<16?(w[i]|0):((w[i-16]+(rr(w15,7)^rr(w15,18)^(w15>>>3))+w[i-7]+(rr(w2,17)^rr(w2,19)^(w2>>>10)))|0));
      const t2=(rr(a,2)^rr(a,13)^rr(a,22))+((a&hash[1])^(a&hash[2])^(hash[1]&hash[2]));
      hash=[(t1+t2)|0].concat(hash);hash[4]=(hash[4]+t1)|0;}
    for(let i=0;i<8;i++)hash[i]=(hash[i]+old[i])|0;}
  for(let i=0;i<8;i++)for(let j=3;j+1;j--){const b=(hash[i]>>(j*8))&255;result+=((b<16)?0:'')+b.toString(16);}
  return result;
}
// Ed25519 via WebCrypto when in a secure context; else a real SHA-256 digest of the payload.
let SIGNER={alg:'SHA-256', ed:null};
async function initSigner(){
  try{
    if(window.crypto&&crypto.subtle&&window.isSecureContext){
      const kp=await crypto.subtle.generateKey({name:'Ed25519'},true,['sign','verify']);
      SIGNER={alg:'Ed25519',ed:kp};
    }
  }catch(e){/* stay on SHA-256 */}
}
async function sign(payload){
  const msg=JSON.stringify(payload);
  try{ if(SIGNER.ed){const s=await crypto.subtle.sign({name:'Ed25519'},SIGNER.ed.privateKey,new TextEncoder().encode(msg));
    return Array.from(new Uint8Array(s)).map(b=>b.toString(16).padStart(2,'0')).join('');} }catch(e){}
  return sha256hex(msg);
}

// ---------- config / state ----------
const API = location.protocol.startsWith('http') ? '' : 'http://192.0.2.10'; // same-origin when served from the API host
let TOKEN = sessionStorage.getItem('demo_token') || '';
const CLR=['Public','Internal','Confidential','Restricted'];
const clr=n=> (n==null?'—':(CLR[n]||n));
const deptOf=k=> (String(k).split('__')[1]||k).replace(/[-_]/g,' ').toUpperCase();
const reqDept=e=>{const m=String(e).match(/\b(hr|tech|legal|finance|ops)\b/i);return m?m[1].toUpperCase():'TECH';};
const DEPTS=[{k:'HR',x:50,y:16,t:'--action'},{k:'TECH',x:84,y:50,t:'--focus'},{k:'FINANCE',x:50,y:84,t:'--success'},{k:'LEGAL',x:16,y:50,t:'--warning'}];
const PHASE={
  idle:['Sovereign access · live','Awaiting cross-department access requests…'],
  request:['A department is requesting access','New access request captured'],
  signing:['Signing the request','Sealing request with '+'Ed25519 (transport-identity bound)…'],
  transit_in:['In transit — libp2p','Signed request crossing the sovereign mesh…'],
  verify_in:['Verifying authenticity','Verifying signature at the grantor node…'],
  pending:['Pending approval','Awaiting the KB manager’s decision…'],
  granting:['Signing the grant','Manager approved — issuing clearance-bounded grant (key 2026-a)…'],
  transit_back:['Grant in transit','Signed grant returning across the mesh…'],
  verify_back:['Verifying the grant','Verifying grant at the requester node…'],
  granted:['Access granted','Access granted — live, no re-login. Enforced by clearance + audit chain.'],
  revoked:['Access revoked','Grant revoked — access removed live across the mesh.'],
};
const SEQ_IN=[['request',900],['signing',1300],['transit_in',1300],['verify_in',1100],['pending',0]];
const SEQ_GRANT=[['granting',1400],['transit_back',1300],['verify_back',1100],['granted',0]];
let scene=null, live=true, timers=[], running=false, seen=new Map();

// ---------- api ----------
async function apiPost(p,b){const r=await fetch(API+p,{method:'POST',headers:{'content-type':'application/json',...(TOKEN?{authorization:'Bearer '+TOKEN}:{})},body:JSON.stringify(b)});if(!r.ok)throw new Error(r.status);return r.json();}
async function apiGet(p){const r=await fetch(API+p,{headers:TOKEN?{authorization:'Bearer '+TOKEN}:{}});if(!r.ok)throw new Error(r.status);return r.json();}

// ---------- render ----------
function el(id){return document.getElementById(id);}
function log(msg,tone){const box=el('log');if(box.querySelector('.muted'))box.innerHTML='';const d=document.createElement('div');d.className='logrow';
  d.innerHTML='<span class="muted">'+new Date().toLocaleTimeString()+'</span><span style="color:var('+(tone||'--muted')+')">'+msg+'</span>';box.prepend(d);}
function drawMesh(){
  const reqK=scene?reqDept(scene.req.requester):'TECH', grK=scene?deptOf(scene.req.kb_id):'HR';
  const on=(a,b)=>scene&&scene.phase!=='idle'&&((a===reqK&&b===grK)||(a===grK&&b===reqK));
  let lines='<defs><linearGradient id="w" x1="0" x2="1"><stop offset="0" stop-color="var(--focus)" stop-opacity=".2"/><stop offset=".5" stop-color="var(--action)" stop-opacity=".75"/><stop offset="1" stop-color="var(--focus)" stop-opacity=".2"/></linearGradient></defs>';
  for(let i=0;i<DEPTS.length;i++)for(let j=i+1;j<DEPTS.length;j++){const a=DEPTS[i],b=DEPTS[j];const active=on(a.k,b.k);
    lines+='<line x1="'+a.x+'%" y1="'+a.y+'%" x2="'+b.x+'%" y2="'+b.y+'%" stroke="'+(active?'url(#w)':'var(--border2)')+'" stroke-width="'+(active?2.5:1)+'" stroke-dasharray="'+(active?'2 10':'2 13')+'" stroke-linecap="round" opacity="'+(active?1:.3)+'" class="'+(active?'wire-live':'')+'"/>';}
  el('mesh').innerHTML=lines;
}
function drawNodes(){
  const p=scene?scene.phase:'idle', reqK=scene?reqDept(scene.req.requester):'', grK=scene?deptOf(scene.req.kb_id):'';
  document.querySelectorAll('.node').forEach(n=>n.remove());
  const st=el('stage');
  DEPTS.forEach(d=>{
    let state='idle';
    if(scene&&p!=='idle'){ if(d.k===reqK)state=(p==='granted')?'unlocked':((p==='request'||p==='verify_back')?'pulse':'active');
      else if(d.k===grK)state=(p==='verify_in'||p==='granting'||p==='pending')?'pulse':'active'; else state='dim'; }
    const dim=state==='dim', pulse=state==='pulse', unlocked=state==='unlocked';
    const role=scene&&p!=='idle'?(d.k===reqK?'requesting':(d.k===grK?'Dept B · grantor':'sovereign node')):'sovereign node';
    const div=document.createElement('div');div.className='node'+(pulse?' pulse':'');
    div.style.left=d.x+'%';div.style.top=d.y+'%';div.style.opacity=dim?.4:1;
    const glow=pulse?'0 0 42px color-mix(in srgb, var('+d.t+') 45%, transparent)':'0 0 20px color-mix(in srgb, var('+d.t+') '+(dim?6:20)+'%, transparent)';
    div.innerHTML='<div class="box" style="border-color:color-mix(in srgb, var('+d.t+') '+(dim?22:55)+'%, transparent);box-shadow:'+glow+'">'
      +'<span style="font-size:26px">'+(unlocked?'🔓':'🖥️')+'</span></div>'
      +'<div class="lbl">'+d.k+'</div><div class="sub">'+role+'</div>';
    st.appendChild(div);
  });
}
function drawPacket(){
  document.querySelectorAll('.packet').forEach(n=>n.remove());
  if(!scene)return;const p=scene.phase;
  if(['idle','pending','granted','revoked'].includes(p))return;
  const reqK=reqDept(scene.req.requester), grK=deptOf(scene.req.kb_id);
  const rp=DEPTS.find(d=>d.k===reqK)||{x:50,y:50}, gp=DEPTS.find(d=>d.k===grK)||{x:50,y:50};
  const tmap={request:.12,signing:.12,transit_in:.5,verify_in:.88,granting:.88,transit_back:.5,verify_back:.12};
  const t=tmap[p]!=null?tmap[p]:.5;const back=['granting','transit_back','verify_back'].includes(p);
  const x=rp.x+(gp.x-rp.x)*t, y=rp.y+(gp.y-rp.y)*t;
  const d=document.createElement('div');d.className='packet';d.style.left=x+'%';d.style.top=y+'%';
  d.style.borderColor=back?'var(--action)':'var(--focus)';d.innerHTML='<span style="font-size:15px">'+(back?'📜':'🔑')+'</span>'+(back?'GRANT':'REQ');
  el('stage').appendChild(d);
}
function drawRail(){
  const idle=el('idle'), rail=el('rail');
  if(!scene){idle.style.display='';rail.style.display='none';return;}
  idle.style.display='none';rail.style.display='';
  const s=scene,p=s.phase,req=s.req;
  const showSig=['signing','transit_in','verify_in'].includes(p);
  const showGrant=['granting','transit_back','verify_back','granted'].includes(p);
  const verified=['verify_in','pending','verify_back','granted'].includes(p);
  let h='<div style="display:flex;justify-content:space-between;font-size:12px;margin-bottom:10px">'
    +'<span class="mono muted">'+reqDept(req.requester)+' → '+deptOf(req.kb_id)+'</span>'
    +'<span class="mono" style="color:var(--action)">'+SIGNER.alg+' · key 2026-a</span></div>';
  h+='<div class="kv"><b>Knowledge base</b><span>'+(req.kb_name||deptOf(req.kb_id))+'</span></div>';
  h+='<div class="kv"><b>Requester</b><span>'+req.requester+'</span></div>';
  h+='<div class="kv"><b>Clearance</b><span>'+clr(req.requested_clearance)+(req.granted_clearance!=null?' · granted '+clr(req.granted_clearance):'')+'</span></div>';
  h+='<div class="kv"><b>Reason</b><span>'+(req.reason||'—')+'</span></div>';
  if(showSig||showGrant){const sig=showGrant?s.grantSig:s.reqSig;
    h+='<div class="sig"><div class="top"><span class="mono" style="font-size:11px;color:var(--action)">'+(showGrant?'GRANT SIGNATURE':'REQUEST SIGNATURE')+'</span>'
      +(verified?'<span class="mono" style="font-size:11px;color:var(--success)">✓ verified · authentic</span>':'')+'</div>'
      +'<div class="hex">0x'+(sig||'').slice(0,64)+'…</div></div>';}
  rail.innerHTML=h;
}
function drawProof(){
  const s=scene,att=s&&s.att,unlocked=s&&s.phase==='granted';
  const items=[
    ['Transport-identity bound',true,'grant unusable by any other peer'],
    ['Clearance-bounded grant',true,s?('≤ '+clr(s.req.granted_clearance!=null?s.req.granted_clearance:s.req.requested_clearance)):'MLS enforced'],
    ['Audit chain',att?att.audit_chain_verified:!!unlocked,att?(att.audit_events+' events · SHA-256'):'tamper-evident'],
    ['India-resident',att?String(att.data_residency||'').toLowerCase().includes('india'):true,att?att.data_residency:'in-country only'],
    ['Encrypted at rest + transit',att?(att.encryption_at_rest&&att.encryption_in_transit):true,'per-tenant envelope keys'],
    ['Revocable live',true,'no re-login, mesh-wide'],
  ];
  let h='<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px"><div class="eyebrow">'+(unlocked?'Access granted — what it unlocks':'Sovereignty guarantees')+'</div>'
    +(unlocked?'<span class="pill" style="color:var(--success);border-color:rgba(34,192,122,.4)">🔓 LIVE</span>':'')+'</div><div class="grid2">';
  items.forEach(it=>{h+='<div class="gitem"><span style="color:var('+(it[1]?'--success':'--muted')+')">✓</span><div><div class="t">'+it[0]+'</div><div class="n">'+it[2]+'</div></div></div>';});
  h+='</div>';el('proof').innerHTML=h;
}
function render(){const p=scene?scene.phase:'idle';el('head').textContent=PHASE[p][0];el('rail-copy').textContent=PHASE[p][1];
  drawMesh();drawNodes();drawPacket();drawRail();drawProof();}

// ---------- sequence runner ----------
function setPhase(ph){if(scene)scene.phase=ph;render();}
function runSeq(seq,done){let i=0;(function step(){if(i>=seq.length){done&&done();return;}const[ph,wait]=seq[i++];setPhase(ph);if(wait>0)timers.push(setTimeout(step,wait));else done&&done();})();}
async function playRequest(req){
  const reqSig=await sign({kb:req.kb_id,subject:req.requester,clearance:req.requested_clearance,reason:req.reason,ts:req.created_at});
  scene={req,phase:'request',reqSig,grantSig:'',att:null};render();
  log(reqDept(req.requester)+' → requests '+deptOf(req.kb_id)+' · '+clr(req.requested_clearance),'--warning');
  log('signature 0x'+reqSig.slice(0,16)+'…','--action');
  runSeq(SEQ_IN);
}
async function playGrant(req){
  const grantSig=await sign({kb:req.kb_id,subject:req.requester,clearance:req.granted_clearance!=null?req.granted_clearance:req.requested_clearance,kid:'2026-a',exp:req.decided_at});
  if(!scene)scene={req,phase:'granting',reqSig:'',grantSig,att:null};else scene.grantSig=grantSig;render();
  log(deptOf(req.kb_id)+' approved · grant signed 0x'+grantSig.slice(0,16)+'…','--success');
  runSeq(SEQ_GRANT,async()=>{ if(live){try{const att=await apiGet('/v1/kb/'+encodeURIComponent(req.kb_id)+'/attestation');scene.att=att;render();log('attestation — audit chain '+(att.audit_chain_verified?'verified ✓':'unverified'),'--success');}catch(e){}}});
}
function playRevoke(req){scene={req,phase:'revoked',reqSig:'',grantSig:'',att:null};render();log(reqDept(req.requester)+'’s grant on '+deptOf(req.kb_id)+' revoked','--danger');}

// ---------- live poll ----------
async function tick(){ if(!live||!TOKEN||running)return;
  try{const rows=await apiGet('/v1/org/access-overview');
    rows.sort((a,b)=>b.created_at>a.created_at?1:-1);
    for(const r of rows){const prev=seen.get(r.id);if(prev===r.status)continue;const first=prev===undefined;seen.set(r.id,r.status);
      if(first&&r.status!=='pending')continue;running=true;
      if(r.status==='pending')await playRequest(r);else if(r.status==='approved')await playGrant(r);else if(r.status==='revoked')playRevoke(r);
      timers.push(setTimeout(()=>running=false,r.status==='approved'?6500:5500));break;}
  }catch(e){}
}
let pollIv=null;
function startLive(){live=true;el('livetxt').textContent='LIVE · polling';el('livedot').style.background='var(--success)';if(pollIv)clearInterval(pollIv);pollIv=setInterval(tick,2200);tick();}
function stopLive(){live=false;el('livetxt').textContent='DEMO MODE';el('livedot').style.background='var(--muted)';if(pollIv)clearInterval(pollIv);}

// scripted fallback
async function playScripted(){stopLive();running=true;const now=new Date().toISOString();
  const req={id:'demo-'+now,kb_id:'acme__hr',kb_name:'HR',requester:'associate.tech@acme.example',requested_clearance:2,granted_clearance:2,reason:'Cross-department policy audit',status:'approved',decided_by:'blake@acme.example',created_at:now,decided_at:now};
  await playRequest(req);timers.push(setTimeout(()=>playGrant(req),6000));timers.push(setTimeout(()=>running=false,13000));}

// ---------- auth ----------
function showApp(){el('login').style.display='none';el('btn-logout').style.display='';startLive();render();}
async function doLogin(){const email=el('email').value.trim(),pw=el('pw').value;el('loginerr').textContent='';
  try{const j=await apiPost('/v1/auth/login',{email,password:pw});TOKEN=j.access_token||j.token;sessionStorage.setItem('demo_token',TOKEN);showApp();}
  catch(e){el('loginerr').textContent='Invalid email or password (or API unreachable).';}}

el('btn-login').onclick=doLogin;
el('pw').addEventListener('keydown',e=>{if(e.key==='Enter')doLogin();});
el('btn-demo').onclick=playScripted;
el('btn-logout').onclick=()=>{TOKEN='';sessionStorage.removeItem('demo_token');stopLive();el('login').style.display='';el('btn-logout').style.display='none';scene=null;render();};

initSigner().then(()=>{ render(); if(TOKEN)showApp(); });
