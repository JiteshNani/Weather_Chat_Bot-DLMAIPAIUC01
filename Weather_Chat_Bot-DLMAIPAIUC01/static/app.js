const chat = document.getElementById("chat");
const form = document.getElementById("chatForm");
const msgInput = document.getElementById("msg");
const locBtn = document.getElementById("locBtn");

function addMessage(text, who="bot"){
  const div = document.createElement("div");
  div.className = `msg ${who}`;
  div.innerText = text;
  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;
}

async function sendMessage(message, extra={}){
  addMessage(message, "user");
  msgInput.value = "";
  const res = await fetch("/chat", {
    method:"POST",
    headers:{ "Content-Type":"application/json" },
    body: JSON.stringify({ message, ...extra })
  });
  const data = await res.json();
  addMessage(data.reply || "No reply received.", "bot");
}

form.addEventListener("submit", (e)=>{
  e.preventDefault();
  const message = msgInput.value.trim();
  if(!message) return;
  sendMessage(message);
});

locBtn.addEventListener("click", ()=>{
  if(!navigator.geolocation){
    addMessage("Geolocation is not supported in this browser.", "bot");
    return;
  }
  addMessage("Getting your location…", "bot");
  navigator.geolocation.getCurrentPosition(
    (pos)=>{
      const lat = pos.coords.latitude;
      const lon = pos.coords.longitude;
      sendMessage("Use my location", {lat, lon});
    },
    ()=>{
      addMessage("I couldn't access your location (permission denied). You can type a city name instead.", "bot");
    },
    { enableHighAccuracy:false, timeout:8000 }
  );
});

// welcome
addMessage("Hi! Ask me about weather anywhere in the world 🌍", "bot");
addMessage("Example: Will it rain in Lisbon tomorrow morning?", "bot");
