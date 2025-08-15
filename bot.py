import asyncio
import json
import os
import re
import time
import random
from typing import Dict, Any, List

import requests
from telegram import Update
from telegram.constants import ParseMode
from telegram.ext import (
    Application, CommandHandler, MessageHandler, ContextTypes,
    filters
)

from config import (
    TELEGRAM_TOKEN, HF_TOKEN, HF_TEXT_MODEL, HF_IMAGE_MODEL,
    MODE, PORT, BASE_URL, DATA_DIR, STATE_PATH, ABILITIES_PATH,
    DELETE_CONFIRM_SECONDS, MODEL_WAIT
)

# ---------- ensure dirs ----------
os.makedirs(DATA_DIR, exist_ok=True)

# ---------- utilities ----------
DICE_REGEX = re.compile(r"(\d*)d(\d+)([+-]\d+)?", re.I)

def parse_args_kv(s: str) -> Dict[str, str]:
    out = {}
    if not s:
        return out
    for k, _, v1, v2 in re.findall(r'(\w+)=("([^"]+)"|(\S+))', s):
        out[k] = v1 or v2
    return out

def tokenize(s: str) -> List[str]:
    return re.findall(r'\S+', s or "")

def roll_once(n: int) -> int:
    return random.randint(1, n)

def roll_expr(expr: str):
    if not expr:
        return None
    m = DICE_REGEX.fullmatch(str(expr).strip())
    if not m:
        return None
    count = int(m.group(1) or 1)
    sides = int(m.group(2))
    mod = int(m.group(3) or 0)
    rolls = [roll_once(sides) for _ in range(count)]
    total = sum(rolls) + mod
    return {"count": count, "sides": sides, "mod": mod, "rolls": rolls, "total": total}

def d20_advdis(mode="normal"):
    r1, r2 = roll_once(20), roll_once(20)
    if mode == "adv":
        return {"rolls": [r1, r2], "pick": max(r1, r2), "mode": "adv"}
    if mode == "dis":
        return {"rolls": [r1, r2], "pick": min(r1, r2), "mode": "dis"}
    return {"rolls": [r1], "pick": r1, "mode": "normal"}

def mod_from_score(score: int) -> int:
    return (int(score) - 10) // 2

# ---------- random generators ----------
PERSONALITIES = [
    "berani", "penakut", "licik", "polos", "keras kepala",
    "misterius", "cerewet", "tenang", "kejam", "dermawan"
]
RACES = ["Human", "Elf", "Dwarf", "Orc", "Halfling", "Tiefling"]
CLASSES = ["Fighter", "Rogue", "Wizard", "Cleric", "Ranger", "Barbarian"]

def rand_stat_4d6_drop1():
    rolls = sorted([roll_once(6) for _ in range(4)], reverse=True)[:3]
    return sum(rolls)

def random_stats():
    # 6 ability scores using 4d6 drop lowest
    return {
        "str": rand_stat_4d6_drop1(),
        "dex": rand_stat_4d6_drop1(),
        "con": rand_stat_4d6_drop1(),
        "int": rand_stat_4d6_drop1(),
        "wis": rand_stat_4d6_drop1(),
        "cha": rand_stat_4d6_drop1(),
    }

def random_personality():
    return random.choice(PERSONALITIES)

def random_npc_suggestion():
    name = random.choice(
        ["Gor", "Lina", "Thar", "Mira", "Eko", "Suri", "Dorn", "Kira", "Tomo", "Nira"]
    ) + random.choice(["", "", "", " the Bold", " of Ash", " Quickhand"])
    race = random.choice(RACES)
    cls = random.choice(CLASSES)
    stats = random_stats()
    base_hp = 8 + mod_from_score(stats["con"])
    ac = 10 + mod_from_score(stats["dex"])
    resist_pool = ["fire", "cold", "poison", "lightning", "necrotic"]
    weak_pool = ["radiant", "bludgeoning", "piercing", "slashing"]
    resist = random.sample(resist_pool, k=random.randint(0, 2))
    weak = random.sample(weak_pool, k=random.randint(0, 2))
    return {
        "name": name,
        "race": race,
        "cls": cls,
        "hp": max(4, base_hp),
        "maxHp": max(4, base_hp),
        "ac": max(10, ac),
        "stats": stats,
        "resist": resist,
        "weak": weak,
        "personality": random_personality(),
        "notes": ""
    }

# ---------- HF helpers ----------
def hf_text(prompt: str, max_new_tokens=280, temperature=0.8) -> str:
    if not HF_TOKEN:
        return "(GM offline: set HUGGING_FACE_API_TOKEN di env)"
    url = f"https://api-inference.huggingface.co/models/{HF_TEXT_MODEL}"
    headers = {"Authorization": f"Bearer {HF_TOKEN}"}
    payload = {
        "inputs": prompt,
        "parameters": {"max_new_tokens": max_new_tokens, "temperature": temperature},
        "options": {"wait_for_model": MODEL_WAIT}
    }
    r = requests.post(url, headers=headers, json=payload, timeout=60)
    if not r.ok:
        return f"(GM gagal: {r.status_code})"
    data = r.json()
    if isinstance(data, list) and data and "generated_text" in data[0]:
        return data[0]["generated_text"]
    if isinstance(data, dict) and "generated_text" in data:
        return data["generated_text"]
    if isinstance(data, str):
        return data
    return json.dumps(data)[:1200]

def hf_image(prompt: str, width=1024, height=1024, steps=28, guidance=7.5):
    if not HF_TOKEN:
        return None, "HF token kosong"
    url = f"https://api-inference.huggingface.co/models/{HF_IMAGE_MODEL}"
    headers = {"Authorization": f"Bearer {HF_TOKEN}", "Accept": "application/octet-stream"}
    payload = {
        "inputs": prompt,
        "parameters": {"width": width, "height": height, "num_inference_steps": steps, "guidance_scale": guidance},
        "options": {"wait_for_model": MODEL_WAIT}
    }
    r = requests.post(url, headers=headers, json=payload, timeout=120)
    if not r.ok:
        try:
            return None, r.text
        except Exception:
            return None, f"HTTP {r.status_code}"
    fname = os.path.join(DATA_DIR, f"img_{int(time.time())}.png")
    with open(fname, "wb") as f:
        f.write(r.content)
    return fname, None

# ---------- state ----------
default_state: Dict[str, Any] = {
    "gameTitle": None,
    "started": False,
    "party": {},     # key: user_id -> char
    "npcs": {},      # key: name -> npc
    "chatLog": [],
    "gmStyle": "GM sinematik, ringkas, Bahasa Indonesia.",
    "worldSeed": "Medieval fantasy di pelabuhan & pasar rempah.",
    "autoGM": False,
    "battle": {
        "inBattle": False,
        "initiative": [],
        "idx": 0,
        "round": 1,
        "combatants": {}  # name -> {side, alive}
    },
    "presets": {}  # akan diisi dari abilities.json saat load
}

pending_deletes: Dict[str, Dict[str, Any]] = {}  # name -> {uid, ts}

state_lock = asyncio.Lock()
state: Dict[str, Any] = {}

def load_json(path: str, fallback):
    try:
        if not os.path.exists(path):
            return fallback
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return fallback

def save_json(path: str, data):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)

def load_state():
    global state
    s = load_json(STATE_PATH, default_state)
    # merge defaults
    merged = json.loads(json.dumps(default_state))
    def deepmerge(a,b):
        for k,v in b.items():
            if isinstance(v, dict):
                a[k] = deepmerge(a.get(k, {}), v)
            else:
                a[k] = v
        return a
    state = deepmerge(merged, s)

def save_state():
    save_json(STATE_PATH, state)

def load_presets():
    presets = load_json(ABILITIES_PATH, {})
    if not isinstance(presets, dict):
        presets = {}
    state["presets"] = presets

def ensure_char(uid: str, name_hint: str = None):
    if uid not in state["party"]:
        # random baseline
        stats = random_stats()
        base_hp = 10 + mod_from_score(stats["con"])
        state["party"][uid] = {
            "name": name_hint or f"Player{uid[-4:]}",
            "race": random.choice(RACES),
            "cls": random.choice(CLASSES),
            "lvl": 1,
            "hp": base_hp,
            "maxHp": base_hp,
            "ac": 10 + mod_from_score(stats["dex"]),
            "stats": stats,
            "inv": [],
            "notes": ""
        }
    return state["party"][uid]

def ensure_npc(name: str):
    if name not in state["npcs"]:
        n = random_npc_suggestion()
        n["name"] = name
        state["npcs"][name] = n
    return state["npcs"][name]

def find_actor(name: str):
    # search by visible name in party, then npc
    for p in state["party"].values():
        if p["name"].lower() == name.lower():
            return p
    for n in state["npcs"].values():
        if n["name"].lower() == name.lower():
            return n
    return None

def short_party():
    arr = list(state["party"].values())
    if not arr: return "Belum ada pemain."
    return " | ".join(f'{c["name"]}(HP:{c["hp"]})' for c in arr)

def short_npcs():
    arr = list(state["npcs"].values())
    if not arr: return "Tidak ada NPC."
    return " | ".join(f'{n["name"]}(HP:{n["hp"]})' for n in arr)

# ---------- battle ----------
def start_battle(csv_names: str):
    npc_names = [x.strip() for x in (csv_names or "").split(",") if x.strip()]
    combatants = {}
    for p in state["party"].values():
        combatants[p["name"]] = {"side":"party", "alive": p["hp"]>0}
    for nm in npc_names:
        n = find_actor(nm) or ensure_npc(nm)
        combatants[n["name"]] = {"side":"npc", "alive": n["hp"]>0}
    entries = []
    for nm in combatants.keys():
        a = find_actor(nm)
        dex = (a.get("stats") or {}).get("dex", 10)
        dex_mod = mod_from_score(dex)
        r = roll_once(20)
        init = r + dex_mod
        entries.append({"name": nm, "r": r, "dexMod": dex_mod, "init": init})
    entries.sort(key=lambda e:(-e["init"], -e["r"]))
    state["battle"] = {
        "inBattle": True,
        "initiative": [e["name"] for e in entries],
        "idx": 0,
        "round": 1,
        "combatants": combatants
    }
    lines = "\n".join(f'{i+1}. {e["name"]} (d20:{e["r"]}{f"+{e['dexMod']}" if e["dexMod"]>=0 else e["dexMod"]}={e["init"]})'
                      for i,e in enumerate(entries))
    return f"⚔️ Battle dimulai! Ronde 1\nInisiatif:\n{lines}\nGiliran: {state['battle']['initiative'][0]}"

def end_battle():
    state["battle"] = {
        "inBattle": False, "initiative": [], "idx": 0, "round": 1, "combatants": {}
    }
    return "🏁 Battle diakhiri."

def current_turn():
    b = state["battle"]
    if not b["inBattle"] or not b["initiative"]: return None
    return b["initiative"][b["idx"]]

def is_alive(name: str):
    a = find_actor(name)
    return a and a["hp"] > 0

def advance_turn():
    b = state["battle"]
    if not b["inBattle"]: return "Tidak dalam battle."
    N = len(b["initiative"])
    attempts = 0
    while True:
        b["idx"] = (b["idx"] + 1) % N
        if b["idx"] == 0: b["round"] += 1
        attempts += 1
        cand = current_turn()
        if is_alive(cand): break
        if attempts > N*2: break
    return f"➡️ Giliran sekarang: {current_turn()} (Ronde {b['round']})"

def adjust_hp(name: str, delta: int):
    a = find_actor(name)
    if not a: return f'Tokoh "{name}" tidak ditemukan.'
    a["hp"] = max(0, min(a.get("maxHp", a["hp"]), a["hp"] + delta))
    if a["hp"] <= 0:
        return f'💀 **{a["name"]} tumbang!** (HP {a["hp"]})'
    return f'HP {a["name"]}: {a["hp"]}/{a.get("maxHp", a["hp"])}'

def apply_resist_weak(tgt: Dict[str,Any], dmg_total: int, dmg_type: str):
    t = (dmg_type or "").lower()
    resist = [str(x).lower() for x in (tgt.get("resist") or [])]
    weak = [str(x).lower() for x in (tgt.get("weak") or [])]
    final = dmg_total
    notes = ""
    if t in weak and t in resist:
        notes = "(weak & resist)"
    elif t in weak:
        final = dmg_total * 2
        notes = "(weak x2)"
    elif t in resist:
        final = dmg_total // 2
        notes = "(resist 1/2)"
    return final, notes

def attack_flow(attacker_name: str, targets: List[str], to_hit_mod=0,
                adv_mode="normal", dmg_expr="1d6", dmg_type="bludgeoning", is_aoe=False):
    att = find_actor(attacker_name)
    if not att: return f'Penyerang "{attacker_name}" tidak ditemukan.'
    if state["battle"]["inBattle"]:
        cur = current_turn()
        if cur and cur.lower() != attacker_name.lower():
            return f'Bukan giliran {attacker_name}. Giliran sekarang: {cur}.'
    if not targets:
        return "Target tidak disebutkan."

    # resolve 'all'
    if len(targets)==1 and targets[0].lower()=="all":
        side = "party" if any(p["name"].lower()==attacker_name.lower() for p in state["party"].values()) else "npc"
        opp = "npc" if side=="party" else "party"
        targets = [n for n,s in state["battle"]["combatants"].items() if s["side"]==opp]

    test = roll_expr(dmg_expr) if dmg_expr else {"total":0,"rolls":[],"mod":0}
    if dmg_expr and not test:
        return f'Format damage tidak valid: "{dmg_expr}". Contoh: 1d6, 2d8+3'

    out_lines = []
    if is_aoe:
        for tn in targets:
            tgt = find_actor(tn)
            if not tgt: out_lines.append(f"{tn}: tidak ditemukan"); continue
            if tgt["hp"]<=0: out_lines.append(f"{tgt['name']} sudah tumbang."); continue
            rr = roll_expr(dmg_expr) if dmg_expr else {"total":0,"rolls":[],"mod":0}
            base = rr["total"]
            final, notes = apply_resist_weak(tgt, base, dmg_type)
            after = adjust_hp(tgt["name"], -final)
            out_lines.append(f"{tgt['name']}: Damage {base} {notes} → **{final}** → {after} (rolls: {rr['rolls']} {('+'+str(rr['mod'])) if rr['mod']>0 else (rr['mod'] or '')})")
        if state["battle"]["inBattle"]:
            out_lines.append(advance_turn())
        return "\n".join(out_lines)
    else:
        for tn in targets:
            tgt = find_actor(tn)
            if not tgt: out_lines.append(f"{tn}: tidak ditemukan"); continue
            if tgt["hp"]<=0: out_lines.append(f"{tgt['name']} sudah tumbang."); continue
            d20 = d20_advdis(adv_mode)
            tohit = d20["pick"] + int(to_hit_mod or 0)
            hit = tohit >= int(tgt.get("ac", 10))
            line = f'{att["name"]} → {tgt["name"]}: To-Hit d20{("(" + ", ".join(map(str,d20["rolls"])) + f")→{d20["pick"]}") if len(d20["rolls"])>1 else "="+str(d20["pick"])}'
            if to_hit_mod: line += f'{("+" if int(to_hit_mod)>=0 else "")}{int(to_hit_mod)}'
            line += f' = **{tohit}** vs AC {tgt.get("ac",10)} → {"HIT" if hit else "MISS"}'
            if not hit:
                out_lines.append(line); continue
            rr = roll_expr(dmg_expr) if dmg_expr else {"total":0,"rolls":[],"mod":0}
            base = rr["total"]
            final, notes = apply_resist_weak(tgt, base, dmg_type)
            after = adjust_hp(tgt["name"], -final)
            line += f'\n💥 Damage: {base} {notes} → **{final}** → {after} (rolls: {rr["rolls"]} {("+"+str(rr["mod"])) if rr["mod"]>0 else (rr["mod"] or "")})'
            out_lines.append(line)
        if state["battle"]["inBattle"]:
            out_lines.append(advance_turn())
        return "\n\n".join(out_lines)

# ---------- GM ----------
def system_prompt():
    return (
        f"Kamu adalah Game Master Dungeons & Dragons (bahasa Indonesia). "
        f"Gaya: {state['gmStyle']}\n"
        f"Dunia: {state['worldSeed']}\n"
        f"Party: {short_party()}\n"
        f"NPC: {short_npcs()}\n"
        f"Balas dengan narasi singkat (<= 8 kalimat)."
    )

def gm_narrate(player_input: str) -> str:
    prompt = system_prompt() + "\nPlayer input:\n" + player_input
    return hf_text(prompt, max_new_tokens=220, temperature=0.8)

# ---------- handlers ----------
HELP_TEXT = (
    "*Perintah utama*\n"
    "/help\n"
    "/newgame [judul]\n"
    "/autogm on|off\n"
    "/join name=Nama [random=true]\n"
    "/sheet [Nama]\n"
    "/set hp=.. ac=.. str=.. dex=.. con=.. int=.. wis=.. cha=.. name=\"Nama\"\n"
    "/inv add \"Item\" | /inv remove \"Item\"\n"
    "/npc add name=Goblin hp=7 ac=13 resist=poison weak=fire\n"
    "/npc sheet Goblin | /npc set Goblin hp=4 ac=12 | /npc remove Goblin | /npc list | /npc random [jumlah]\n\n"
    "/roll NdM(+/-K) | /d20 +5 dc=12\n"
    "/aksi ...  (narasi GM) | /cerita | /story [N]\n\n"
    "*Battle*\n"
    "/battle start Goblin,Wolf | /battle turn | /battle end | /next\n"
    "/attack Attacker Target +X dmg NdM type=fire [adv|dis]\n"
    "/attack Attacker targets=Goblin,Wolf dmg=8d6 type=fire aoe\n"
    "/hp Nama +/-N | /status\n\n"
    "*Campaign*\n"
    "/campaign save Nama | /campaign load Nama | /campaign list\n"
    "/campaign delete Nama  (butuh konfirmasi)\n"
    "/campaign confirm-delete Nama\n\n"
    "*Abilities*\n"
    "/ability list\n"
    "/ability add name=Name dmg=1d8|heal=2d8 type=... aoe=true|false desc=\"...\"\n"
    "/ability use Name Attacker targets=Goblin,Wolf [adv|dis]\n\n"
    "*Image*\n"
    "/image prompt=\"...\" type=map|character|monster size=1024\n"
)

async def cmd_help(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(HELP_TEXT, parse_mode=ParseMode.MARKDOWN)

async def cmd_newgame(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    async with state_lock:
        load_state()  # reset to last saved, then override
        state.update(json.loads(json.dumps(default_state)))
        title = " ".join(ctx.args) if ctx.args else "Petualangan Baru"
        state["gameTitle"] = title
        state["started"] = True
        load_presets()
        save_state()
    await update.message.reply_text(f"🎲 {title} dimulai. Pemain daftar: /join name=Nama")

async def cmd_autogm(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    val = (ctx.args[0].lower() if ctx.args else "")
    async with state_lock:
        if val == "on":
            state["autoGM"] = True
        elif val == "off":
            state["autoGM"] = False
        else:
            return await update.message.reply_text("Gunakan: /autogm on|off")
        save_state()
    await update.message.reply_text(f"Auto-GM: {'ON' if state['autoGM'] else 'OFF'}")

async def cmd_join(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    args = parse_args_kv(" ".join(ctx.args))
    uid = str(update.effective_user.id)
    name_hint = args.get("name") or update.effective_user.first_name or f"Player{uid[-4:]}"
    make_random = (args.get("random","false").lower() in ["true","1","yes","y"])
    async with state_lock:
        ch = ensure_char(uid, name_hint)
        if args.get("name"): ch["name"] = args["name"]
        if make_random:
            st = random_stats()
            ch["stats"] = st
            ch["hp"] = 10 + mod_from_score(st["con"])
            ch["maxHp"] = ch["hp"]
            ch["ac"] = 10 + mod_from_score(st["dex"])
            ch["race"] = random.choice(RACES)
            ch["cls"] = random.choice(CLASSES)
        # overrides
        for k in ["hp","ac","lvl"]:
            if k in args: ch[k] = int(args[k])
        save_state()
    await update.message.reply_text(f"✅ Bergabung: {ch['name']} | HP {ch['hp']}/{ch['maxHp']} | AC {ch['ac']}")

async def cmd_sheet(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    who = " ".join(ctx.args).strip() if ctx.args else ""
    async with state_lock:
        if not who:
            ch = ensure_char(str(update.effective_user.id), update.effective_user.first_name)
        else:
            ch = find_actor(who)
        if not ch:
            return await update.message.reply_text("Karakter tidak ditemukan.")
        inv = ", ".join(ch.get("inv", [])) or "-"
        txt = (f"📜 {ch['name']}\nHP {ch['hp']}/{ch.get('maxHp', ch['hp'])} | AC {ch.get('ac',10)}\n"
               f"STR:{ch['stats'].get('str',10)} DEX:{ch['stats'].get('dex',10)} CON:{ch['stats'].get('con',10)}\n"
               f"INT:{ch['stats'].get('int',10)} WIS:{ch['stats'].get('wis',10)} CHA:{ch['stats'].get('cha',10)}\n"
               f"Inventory: {inv}\nNotes: {ch.get('notes','-')}")
    await update.message.reply_text(txt)

async def cmd_set(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    args = parse_args_kv(" ".join(ctx.args))
    uid = str(update.effective_user.id)
    async with state_lock:
        ch = ensure_char(uid, update.effective_user.first_name)
        mapping = {"name":"name","hp":"hp","maxhp":"maxHp","ac":"ac","lvl":"lvl"}
        changed = []
        for k,v in args.items():
            if k in ["str","dex","con","int","wis","cha"]:
                ch["stats"][k] = int(v); changed.append(f"{k}={v}")
            elif k in mapping:
                ch[mapping[k]] = (v if mapping[k]=="name" else int(v)); changed.append(f"{k}={v}")
        save_state()
    await update.message.reply_text("🛠 Diubah: " + (", ".join(changed) if changed else "(tidak ada)"))

async def cmd_inv(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not ctx.args: return await update.message.reply_text('Gunakan: /inv add "Item" | /inv remove "Item"')
    sub = ctx.args[0].lower()
    content = " ".join(ctx.args[1:])
    item = re.sub(r'^"(.+)"$', r"\1", content).strip()
    uid = str(update.effective_user.id)
    async with state_lock:
        ch = ensure_char(uid, update.effective_user.first_name)
        if sub == "add" and item:
            ch["inv"].append(item); save_state()
            return await update.message.reply_text(f"➕ {item} ditambahkan.")
        if sub == "remove" and item:
            try:
                idx = next(i for i,x in enumerate(ch["inv"]) if x.lower()==item.lower())
                ch["inv"].pop(idx); save_state()
                return await update.message.reply_text(f"➖ {item} dihapus.")
            except StopIteration:
                return await update.message.reply_text(f'Item "{item}" tidak ditemukan.')
    await update.message.reply_text('Gunakan: /inv add "Item" | /inv remove "Item"')

async def cmd_npc(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not ctx.args:
        return await update.message.reply_text('Gunakan: /npc add|sheet|set|remove|list|random [jumlah]')
    sub = ctx.args[0].lower()
    async with state_lock:
        if sub == "add":
            args = parse_args_kv(" ".join(ctx.args[1:]))
            if "name" not in args:
                return await update.message.reply_text('Format: /npc add name=Goblin hp=7 ac=13 resist=fire weak=poison')
            npc = ensure_npc(args["name"])
            if "hp" in args: npc["hp"] = int(args["hp"]); npc["maxHp"] = max(npc.get("maxHp",0), npc["hp"])
            if "ac" in args: npc["ac"] = int(args["ac"])
            for s in ["str","dex","con","int","wis","cha"]:
                if s in args: npc["stats"][s] = int(args[s])
            if "resist" in args: npc["resist"] = [x.strip() for x in str(args["resist"]).split(",") if x.strip()]
            if "weak" in args: npc["weak"] = [x.strip() for x in str(args["weak"]).split(",") if x.strip()]
            if "personality" in args: npc["personality"] = args["personality"]
            save_state()
            return await update.message.reply_text(
                f"NPC {npc['name']} siap. Resist:[{', '.join(npc.get('resist',[])) or '-'}] Weak:[{', '.join(npc.get('weak',[])) or '-'}]\nKepribadian: {npc.get('personality','-')}"
            )

        if sub == "sheet":
            name = " ".join(ctx.args[1:]).strip()
            n = state["npcs"].get(name) or find_actor(name)
            if not n or n["name"] not in state["npcs"]:
                return await update.message.reply_text(f"NPC {name} tidak ditemukan.")
            txt = (f"📜 {n['name']}\nHP {n['hp']}/{n['maxHp']} | AC {n['ac']}\n"
                   f"Resist: [{', '.join(n.get('resist',[])) or '-'}] Weak: [{', '.join(n.get('weak',[])) or '-'}]\n"
                   f"Kepribadian: {n.get('personality','-')}")
            return await update.message.reply_text(txt)

        if sub == "set":
            if len(ctx.args) < 2:
                return await update.message.reply_text('Format: /npc set Goblin hp=4 ac=12')
            name = ctx.args[1]
            n = state["npcs"].get(name) or find_actor(name)
            if not n or n["name"] not in state["npcs"]:
                return await update.message.reply_text(f"NPC {name} tidak ditemukan.")
            args = parse_args_kv(" ".join(ctx.args[2:]))
            if "hp" in args: n["hp"] = int(args["hp"])
            if "maxhp" in args: n["maxHp"] = int(args["maxhp"])
            if "ac" in args: n["ac"] = int(args["ac"])
            for s in ["str","dex","con","int","wis","cha"]:
                if s in args: n["stats"][s] = int(args[s])
            if "resist" in args: n["resist"] = [x.strip() for x in str(args["resist"]).split(",") if x.strip()]
            if "weak" in args: n["weak"] = [x.strip() for x in str(args["weak"]).split(",") if x.strip()]
            if "personality" in args: n["personality"] = args["personality"]
            save_state()
            return await update.message.reply_text(f"NPC {n['name']} diupdate.")

        if sub == "remove":
            name = " ".join(ctx.args[1:]).strip()
            if not name: return await update.message.reply_text('Gunakan: /npc remove Goblin')
            if name not in state["npcs"]: return await update.message.reply_text(f"NPC {name} tidak ditemukan.")
            del state["npcs"][name]; save_state()
            return await update.message.reply_text(f"NPC {name} dihapus.")

        if sub == "list":
            return await update.message.reply_text(short_npcs())

        if sub == "random":
            count = int(ctx.args[1]) if len(ctx.args)>1 and ctx.args[1].isdigit() else 1
            made = []
            for _ in range(max(1, min(count, 10))):
                npc = random_npc_suggestion()
                state["npcs"][npc["name"]] = npc
                made.append(npc["name"])
            save_state()
            return await update.message.reply_text("NPC acak dibuat: " + ", ".join(made))

    await update.message.reply_text('Gunakan: /npc add|sheet|set|remove|list|random [jumlah]')

async def cmd_roll(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    expr = " ".join(ctx.args) if ctx.args else "1d20"
    r = roll_expr(expr)
    if not r:
        return await update.message.reply_text('Format: /roll NdM(+/-K). Contoh: /roll 2d6+1')
    sign = f'+{r["mod"]}' if r["mod"]>0 else (str(r["mod"]) if r["mod"] else "")
    txt = f'🎲 Roll {r["count"]}d{r["sides"]}{sign}\nHasil: {r["rolls"]} {sign}\nTotal: {r["total"]}'
    await update.message.reply_text(txt)

async def cmd_d20(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    text = " ".join(ctx.args)
    dc_m = re.search(r"dc\s*=\s*(\d+)", text, re.I)
    mod_m = re.search(r"([+-]\d+)", text)
    mod = int(mod_m.group(1)) if mod_m else 0
    r = roll_once(20)
    total = r + mod
    out = f"🎯 d20: [{r}] {f'+{mod}' if mod>0 else (mod or '')} = {total}"
    if dc_m:
        dc = int(dc_m.group(1))
        out += f" vs DC {dc} → {'SUKSES' if total>=dc else 'GAGAL'}"
    await update.message.reply_text(out)

async def cmd_aksi(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    user_action = update.effective_message.text.partition(" ")[2] or "(tanpa aksi)"
    async with state_lock:
        out = gm_narrate(user_action)
        state["chatLog"].append({"speaker": update.effective_user.first_name, "text": user_action, "ts": int(time.time())})
        state["chatLog"].append({"speaker": "GM", "text": out, "ts": int(time.time())})
        save_state()
    await update.message.reply_text(out)

async def cmd_cerita(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    async with state_lock:
        out = gm_narrate("Lanjutkan adegan berikutnya.")
        state["chatLog"].append({"speaker": "GM", "text": out, "ts": int(time.time())})
        save_state()
    await update.message.reply_text(out)

async def cmd_story(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    n = int(ctx.args[0]) if ctx.args and ctx.args[0].isdigit() else 5
    async with state_lock:
        logs = state["chatLog"][-n:]
    lines = []
    for item in logs:
        lines.append(f"{item['speaker']}: {item['text']}")
    await update.message.reply_text("\n".join(lines) if lines else "(belum ada cerita)")

async def cmd_battle(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not ctx.args:
        return await update.message.reply_text('Gunakan: /battle start NPC1,NPC2 | /battle turn | /battle end')
    sub = ctx.args[0].lower()
    async with state_lock:
        if sub == "start":
            csv = " ".join(ctx.args)[len("start"):].strip()
            msg = start_battle(csv); save_state()
            return await update.message.reply_text(msg)
        if sub == "turn":
            if not state["battle"]["inBattle"]:
                return await update.message.reply_text("Tidak sedang battle.")
            order = " → ".join(state["battle"]["initiative"])
            return await update.message.reply_text(f"Ronde {state['battle']['round']} | Giliran: {current_turn()}\nUrutan: {order}")
        if sub == "end":
            msg = end_battle(); save_state()
            return await update.message.reply_text(msg)
    await update.message.reply_text('Gunakan: /battle start|turn|end')

async def cmd_next(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    async with state_lock:
        if not state["battle"]["inBattle"]:
            return await update.message.reply_text("Tidak dalam battle.")
        msg = advance_turn(); save_state()
    await update.message.reply_text(msg)

async def cmd_attack(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    text = " ".join(ctx.args)
    toks = tokenize(text)
    if not toks:
        return await update.message.reply_text('Format: /attack Attacker Target +X dmg 1d8 type=fire OR /attack Attacker targets=Goblin,Wolf dmg=8d6 type=fire aoe')
    attacker = toks[0]
    kv = parse_args_kv(text)
    targets = []
    if "targets" in kv:
        targets = [x.strip() for x in kv["targets"].split(",") if x.strip()]
    elif len(toks)>1:
        targets = [x.strip() for x in toks[1].split(",") if x.strip()]
    dmg_expr = kv.get("dmg") or next((t for t in toks if DICE_REGEX.fullmatch(t)), "1d6")
    dmg_type = kv.get("type","bludgeoning")
    tohit = int(kv["tohit"]) if "tohit" in kv else (int(re.search(r"([+-]\d+)", text).group(1)) if re.search(r"([+-]\d+)", text) else 0)
    adv_mode = "adv" if re.search(r"\badv\b", text, re.I) else ("dis" if re.search(r"\bdis\b", text, re.I) else "normal")
    is_aoe = bool(kv.get("aoe")) or bool(re.search(r"\baoe\b", text, re.I))
    async with state_lock:
        out = attack_flow(attacker, targets, tohit, adv_mode, dmg_expr, dmg_type, is_aoe)
        save_state()
    await update.message.reply_text(out, parse_mode=ParseMode.MARKDOWN)

async def cmd_hp(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    m = re.match(r"(.+)\s+([+-]\d+)$", " ".join(ctx.args) if ctx.args else "")
    if not m:
        return await update.message.reply_text('Format: /hp Nama +/-N  (contoh: /hp Goblin -4)')
    name, delta = m.group(1).strip(), int(m.group(2))
    async with state_lock:
        msg = adjust_hp(name, delta); save_state()
    await update.message.reply_text(msg, parse_mode=ParseMode.MARKDOWN)

async def cmd_status(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    async with state_lock:
        party = "\n".join(f'{c["name"]} HP:{c["hp"]}/{c["maxHp"]} AC:{c["ac"]}' for c in state["party"].values()) or "-"
        npcs = "\n".join(f'{n["name"]} HP:{n["hp"]}/{n["maxHp"]} AC:{n["ac"]} Resist:[{", ".join(n.get("resist",[])) or "-"}] Weak:[{", ".join(n.get("weak",[])) or "-"}]'
                         for n in state["npcs"].values()) or "-"
        battle = f'Battle: Ronde {state["battle"]["round"]}, Giliran {current_turn()}\nUrutan: {" → ".join(state["battle"]["initiative"])}' if state["battle"]["inBattle"] else "Battle: (tidak aktif)"
    await update.message.reply_text(f"🧙‍♀️ PARTY:\n{party}\n\n👹 NPC/MONSTER:\n{npcs}\n\n{battle}")

async def cmd_image(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    args_text = " ".join(ctx.args)
    kv = parse_args_kv(args_text)
    raw_prompt = kv.get("prompt", args_text) or "fantasy scene, high detail"
    type_ = (kv.get("type","scene")).lower()
    size = int(kv.get("size", "1024"))
    enhanced = raw_prompt
    if type_ == "map":
        enhanced = f"Top-down dungeon map, grid overlay, labeled rooms, doors, corridors. {raw_prompt}"
    elif type_ == "character":
        enhanced = f"Character portrait, waist-up, cinematic lighting, detailed face, fantasy clothing. {raw_prompt}"
    elif type_ == "monster":
        enhanced = f"Detailed monster illustration, dynamic pose, high detail. {raw_prompt}"
    await update.message.reply_text("Membuat gambar... ⏳")
    path, err = hf_image(enhanced, width=size, height=size)
    if err or not path:
        return await update.message.reply_text("Gagal membuat gambar: " + (err or "unknown"))
    with open(path, "rb") as f:
        await update.message.reply_photo(f, caption=f"Gambar: {type_}\nPrompt: {raw_prompt}")

async def cmd_campaign(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not ctx.args:
        return await update.message.reply_text('Gunakan: /campaign save|load|list|delete|confirm-delete Nama')
    sub = ctx.args[0].lower()
    if sub == "save":
        name = " ".join(ctx.args[1:]).strip()
        if not name: return await update.message.reply_text("Gunakan: /campaign save Nama")
        path = os.path.join(DATA_DIR, f"{re.sub(r'\\s+','_',name)}.json")
        async with state_lock:
            save_json(path, state)
        return await update.message.reply_text(f"Campaign disimpan: {name} → {path}")
    if sub == "load":
        name = " ".join(ctx.args[1:]).strip()
        if not name: return await update.message.reply_text("Gunakan: /campaign load Nama")
        path = os.path.join(DATA_DIR, f"{re.sub(r'\\s+','_',name)}.json")
        if not os.path.exists(path): return await update.message.reply_text(f"Campaign tidak ditemukan: {name}")
        async with state_lock:
            data = load_json(path, default_state)
            # hard-merge dengan default
            merged = json.loads(json.dumps(default_state))
            def dm(a,b):
                for k,v in b.items():
                    if isinstance(v, dict): a[k]=dm(a.get(k,{}), v)
                    else: a[k]=v
                return a
            dm(merged, data)
            state.update(merged)
            save_state()
        return await update.message.reply_text(f'Campaign "{name}" dimuat.')
    if sub == "list":
        files = [f[:-5] for f in os.listdir(DATA_DIR) if f.endswith(".json")]
        return await update.message.reply_text("Campaigns:\n" + ("\n".join(files) if files else "(kosong)"))
    if sub == "delete":
        name = " ".join(ctx.args[1:]).strip()
        if not name: return await update.message.reply_text("Gunakan: /campaign delete Nama")
        pending_deletes[name] = {"uid": update.effective_user.id, "ts": int(time.time())}
        return await update.message.reply_text(
            f"⚠️ Permintaan hapus \"{name}\" dibuat.\n"
            f"Kirim /campaign confirm-delete {name} dalam {DELETE_CONFIRM_SECONDS} detik (oleh requester yang sama) untuk menghapus."
        )
    if sub == "confirm-delete":
        name = " ".join(ctx.args[1:]).strip()
        if not name: return await update.message.reply_text("Gunakan: /campaign confirm-delete Nama")
        pd = pending_deletes.get(name)
        if not pd: return await update.message.reply_text(f"Tidak ada permintaan hapus untuk \"{name}\".")
        if pd["uid"] != update.effective_user.id:
            return await update.message.reply_text("Hanya requester yang boleh konfirmasi.")
        if int(time.time()) - pd["ts"] > DELETE_CONFIRM_SECONDS:
            del pending_deletes[name]
            return await update.message.reply_text("Permintaan hapus kedaluwarsa. Ulangi /campaign delete.")
        path = os.path.join(DATA_DIR, f"{re.sub(r'\\s+','_',name)}.json")
        ok = False
        if os.path.exists(path):
            os.remove(path); ok = True
        del pending_deletes[name]
        return await update.message.reply_text("✅ Dihapus." if ok else "Campaign tidak ditemukan.")
    await update.message.reply_text('Perintah campaign tidak dikenali.')

async def cmd_ability(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not ctx.args:
        return await update.message.reply_text('Gunakan: /ability list | /ability add ... | /ability use ...')
    sub = ctx.args[0].lower()
    async with state_lock:
        if sub == "list":
            names = list(state.get("presets", {}).keys())
            return await update.message.reply_text("Presets:\n" + ("\n".join(names) if names else "(kosong)"))
        if sub == "add":
            kv = parse_args_kv(" ".join(ctx.args[1:]))
            if "name" not in kv:
                return await update.message.reply_text('Format: /ability add name=Name dmg=1d8|heal=2d8 type=... aoe=true|false desc="..."')
            name = kv["name"]
            preset = {}
            if "dmg" in kv: preset["dmg"] = kv["dmg"]
            if "heal" in kv: preset["heal"] = kv["heal"]
            if "type" in kv: preset["type"] = kv["type"]
            if "tohit" in kv: preset["toHitMod"] = kv["tohit"]
            if "aoe" in kv: preset["aoe"] = kv["aoe"].lower() in ["true","1","yes","y"]
            if "desc" in kv: preset["desc"] = kv["desc"]
            state["presets"][name] = preset
            save_state()
            # juga update abilities.json supaya persist across new games
            try:
                disk = load_json(ABILITIES_PATH, {})
                disk[name] = preset
                save_json(ABILITIES_PATH, disk)
            except Exception:
                pass
            return await update.message.reply_text(f'Preset "{name}" ditambahkan/diupdate.')
        if sub == "use":
            toks = tokenize(" ".join(ctx.args[1:]))
            if not toks:
                return await update.message.reply_text('Contoh: /ability use Fireball Mado targets=Goblin,Wolf')
            ability_name = toks[0]
            preset = state["presets"].get(ability_name)
            if not preset:
                return await update.message.reply_text(f'Preset "{ability_name}" tidak ditemukan.')
            kv = parse_args_kv(" ".join(ctx.args[1:]))
            attacker = kv.get("attacker") or (toks[1] if len(toks)>1 else None)
            if not attacker:
                return await update.message.reply_text("Sebutkan attacker. Contoh: /ability use Fireball Mado targets=Goblin")
            # targets
            targets = []
            if "targets" in kv:
                targets = [x.strip() for x in kv["targets"].split(",") if x.strip()]
            elif len(toks)>2 and not re.match(r"^(adv|dis|attacker=)", toks[2], re.I):
                if "," in toks[2]: targets = [x.strip() for x in toks[2].split(",")]
                else: targets = [toks[2]]
            # heal
            if "heal" in preset:
                heal_expr = preset["heal"]
                if preset.get("aoe", False) and not targets:
                    # heal semua party
                    targets = [p["name"] for p in state["party"].values()]
                results = []
                for t in targets:
                    tgt = find_actor(t)
                    if not tgt: results.append(f"{t}: tidak ditemukan"); continue
                    r = roll_expr(heal_expr)
                    if not r: results.append(f"Format heal salah: {heal_expr}"); continue
                    amount = r["total"]
                    tgt["hp"] = min(tgt.get("maxHp", tgt["hp"]), tgt["hp"] + amount)
                    results.append(f'{tgt["name"]} sembuh {amount}. HP: {tgt["hp"]}/{tgt.get("maxHp",tgt["hp"])} (rolls: {r["rolls"]} {("+"+str(r["mod"])) if r["mod"]>0 else (r["mod"] or "")})')
                save_state()
                return await update.message.reply_text(f"Ability: {ability_name} ({preset.get('desc','-')})\n\n" + "\n".join(results))
            # damage
            is_aoe = preset.get("aoe", False) or ("aoe" in kv and kv["aoe"].lower() in ["true","1","yes","y"])
            adv_mode = "adv" if re.search(r"\badv\b", " ".join(ctx.args), re.I) else ("dis" if re.search(r"\bdis\b", " ".join(ctx.args), re.I) else "normal")
            tohit = 0
            if preset.get("toHitMod"):
                try:
                    tohit = int(re.search(r"([+-]?\d+)", str(preset["toHitMod"])).group(1))
                except Exception:
                    tohit = 0
            if "tohit" in kv:
                tohit = int(kv["tohit"])
            dmg_expr = preset.get("dmg","1d6")
            dmg_type = preset.get("type","bludgeoning")
            if not targets and not is_aoe:
                return await update.message.reply_text("Target tidak disebut.")
            out = attack_flow(attacker, targets if targets else ["all"], tohit, adv_mode, dmg_expr, dmg_type, is_aoe)
            save_state()
            return await update.message.reply_text(f"Ability: {ability_name} ({preset.get('desc','-')})\n\n{out}", parse_mode=ParseMode.MARKDOWN)
    await update.message.reply_text('Perintah ability tidak dikenali.')

async def text_autogm(update: Update, ctx: ContextTypes.DEFAULT_TYPE):
    if not state.get("autoGM"): return
    msg = update.effective_message.text
    if not msg or msg.startswith("/"): return
    async with state_lock:
        out = gm_narrate(msg)
        state["chatLog"].append({"speaker": update.effective_user.first_name, "text": msg, "ts": int(time.time())})
        state["chatLog"].append({"speaker": "GM", "text": out, "ts": int(time.time())})
        save_state()
    await update.message.reply_text(out)

# ---------- startup ----------
def require_env():
    if not TELEGRAM_TOKEN:
        raise RuntimeError("TELEGRAM_TOKEN kosong. Set env di Render.")
    if not HF_TOKEN:
        print("WARNING: HUGGING_FACE_API_TOKEN kosong. Narasi/gambar tidak aktif.")

def main():
    require_env()
    load_state()
    load_presets()
    app = Application.builder().token(TELEGRAM_TOKEN).build()

    # commands
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("newgame", cmd_newgame))
    app.add_handler(CommandHandler("autogm", cmd_autogm))
    app.add_handler(CommandHandler("join", cmd_join))
    app.add_handler(CommandHandler("sheet", cmd_sheet))
    app.add_handler(CommandHandler("set", cmd_set))
    app.add_handler(CommandHandler("inv", cmd_inv))
    app.add_handler(CommandHandler("npc", cmd_npc))
    app.add_handler(CommandHandler("roll", cmd_roll))
    app.add_handler(CommandHandler("d20", cmd_d20))
    app.add_handler(CommandHandler("aksi", cmd_aksi))
    app.add_handler(CommandHandler("cerita", cmd_cerita))
    app.add_handler(CommandHandler("story", cmd_story))
    app.add_handler(CommandHandler("battle", cmd_battle))
    app.add_handler(CommandHandler("next", cmd_next))
    app.add_handler(CommandHandler("attack", cmd_attack))
    app.add_handler(CommandHandler("hp", cmd_hp))
    app.add_handler(CommandHandler("status", cmd_status))
    app.add_handler(CommandHandler("image", cmd_image))
    app.add_handler(CommandHandler("campaign", cmd_campaign))
    app.add_handler(CommandHandler("ability", cmd_ability))

    # auto-GM text handler
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, text_autogm))

    if MODE == "webhook":
        if not BASE_URL:
            raise RuntimeError("BASE_URL kosong. Set BASE_URL (https://service.onrender.com) di env Render.")
        url_path = TELEGRAM_TOKEN
        webhook_url = f"{BASE_URL.rstrip('/')}/{url_path}"
        print(f"Starting webhook on 0.0.0.0:{PORT} — {webhook_url}")
        app.run_webhook(
            listen="0.0.0.0",
            port=PORT,
            url_path=url_path,
            webhook_url=webhook_url,
        )
    else:
        print("Starting polling mode...")
        app.run_polling()

if __name__ == "__main__":
    main()
