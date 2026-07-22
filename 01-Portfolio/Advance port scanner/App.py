"""
╔══════════════════════════════════════════════════════════════════╗
║       ADVANCED PORT SCANNER  v5.0  —  SUPRAJA TECHNOLOGIES      ║
╚══════════════════════════════════════════════════════════════════╝
INSTALL:   pip install requests reportlab google-generativeai
RUN:       python advanced_port_scanner.py
"""

import os, sys, re, json, csv, socket, platform, threading, webbrowser
import sqlite3, datetime, subprocess, concurrent.futures, ipaddress
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog, colorchooser

# ── optional ───────────────────────────────────────────────────────────────────
try:    import requests;                     REQUESTS_OK = True
except: REQUESTS_OK = False
try:    import ftplib;                       FTP_OK = True
except: FTP_OK = False
try:    import smtplib;                      SMTP_OK = True
except: SMTP_OK = False
try:    import google.generativeai as genai; GEMINI_OK = True
except: GEMINI_OK = False
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.platypus import (SimpleDocTemplate, Paragraph, Table,
                                     TableStyle, Spacer, HRFlowable)
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib import colors as rlc
    from reportlab.lib.units import cm
    REPORTLAB_OK = True
except: REPORTLAB_OK = False

# ══════════════════════════════════════════════════════════════════════════════
#  PATHS  (all relative to ~/Documents/Advanced_Port_Scanner)
# ══════════════════════════════════════════════════════════════════════════════
APP_DIR   = os.path.join(os.path.expanduser("~"), "Documents", "Advanced_Port_Scanner")
SCANS_DIR_DEFAULT = os.path.join(APP_DIR, "scans")
DB_PATH   = os.path.join(APP_DIR, "scanner.db")
CFG_PATH  = os.path.join(APP_DIR, "settings.json")
for d in (APP_DIR, SCANS_DIR_DEFAULT):
    os.makedirs(d, exist_ok=True)

# ══════════════════════════════════════════════════════════════════════════════
#  SETTINGS (persistent JSON)
# ══════════════════════════════════════════════════════════════════════════════
DEFAULT_CFG = {
    "gemini_api_key":  "",
    "scans_folder":    SCANS_DIR_DEFAULT,
    "auto_save_pdf":   True,
    "auto_open_folder":True,
    "assessor_name":   "Security Analyst",
    "default_proto":   "TCP",
    "default_mode":    "Traditional",
    "theme_accent":    "#00ff88",
}

def load_cfg() -> dict:
    if os.path.exists(CFG_PATH):
        try:
            with open(CFG_PATH, encoding="utf-8") as f:
                d = json.load(f)
            # merge any missing keys
            for k, v in DEFAULT_CFG.items():
                d.setdefault(k, v)
            return d
        except: pass
    return dict(DEFAULT_CFG)

def save_cfg(cfg: dict):
    try:
        with open(CFG_PATH, "w", encoding="utf-8") as f:
            json.dump(cfg, f, indent=2)
    except: pass

CFG = load_cfg()

def get_api_key() -> str:
    """Priority: settings.json → environment variable → .env file."""
    if CFG.get("gemini_api_key"):
        return CFG["gemini_api_key"]
    k = os.getenv("GOOGLE_API_KEY", "")
    if k: return k
    env_p = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env")
    if os.path.exists(env_p):
        with open(env_p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line.startswith("GOOGLE_API_KEY="):
                    return line.split("=",1)[1].strip()
    return ""

def scans_dir() -> str:
    p = CFG.get("scans_folder", SCANS_DIR_DEFAULT)
    os.makedirs(p, exist_ok=True)
    return p

# ══════════════════════════════════════════════════════════════════════════════
#  THEME
# ══════════════════════════════════════════════════════════════════════════════
BG         = "#0d1117"
SIDEBAR_BG = "#161b22"
CARD_BG    = "#21262d"
ACCENT     = CFG.get("theme_accent","#00ff88")
ACCENT2    = "#58a6ff"
TEXT       = "#c9d1d9"
TEXT_DIM   = "#8b949e"
RED        = "#f85149"
YELLOW     = "#e3b341"
GREEN      = "#3fb950"
ORANGE     = "#d29922"
BORDER     = "#30363d"
PURPLE     = "#bc8cff"
KNOCK_BG   = "#0a0f1a"
KNOCK_ACC  = "#00d4ff"

SEV_CLR = {"Critical":RED,"High":ORANGE,"Medium":YELLOW,"Low":GREEN,"Info":ACCENT2}

# ══════════════════════════════════════════════════════════════════════════════
#  DATABASE  (scans + chat history)
# ══════════════════════════════════════════════════════════════════════════════
def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""CREATE TABLE IF NOT EXISTS scans(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        scan_name TEXT, target TEXT, ip TEXT,
        port_range TEXT, protocol TEXT,
        open_ports TEXT, vulnerabilities TEXT,
        ai_report TEXT, scan_folder TEXT,
        timestamp TEXT, assessor TEXT)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS chat_history(
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        role TEXT, message TEXT, timestamp TEXT)""")
    existing = {r[1] for r in conn.execute("PRAGMA table_info(scans)")}
    for col in ["scan_name","ip","port_range","protocol","open_ports",
                "vulnerabilities","ai_report","scan_folder","timestamp","assessor"]:
        if col not in existing:
            conn.execute(f"ALTER TABLE scans ADD COLUMN {col} TEXT")
    conn.commit(); conn.close()

def db_save_scan(name,target,ip,pr,proto,rows,vulns,ai,folder,assessor=""):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("INSERT INTO scans(scan_name,target,ip,port_range,protocol,"
                 "open_ports,vulnerabilities,ai_report,scan_folder,timestamp,assessor)"
                 " VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                 (name,target,ip,pr,proto,
                  json.dumps([[r[0],r[1],r[2],r[3],r[4]] for r in rows]),
                  json.dumps(vulns),ai,folder,
                  datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),assessor))
    conn.commit(); conn.close()

def db_all_scans():
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("SELECT id,scan_name,target,ip,port_range,protocol,timestamp"
                        " FROM scans ORDER BY id DESC LIMIT 300").fetchall()
    conn.close(); return rows

def db_one_scan(sid):
    conn = sqlite3.connect(DB_PATH)
    r = conn.execute("SELECT * FROM scans WHERE id=?",(sid,)).fetchone()
    conn.close(); return r

def db_clear_scans():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM scans"); conn.commit(); conn.close()

def db_save_chat(role: str, message: str):
    conn = sqlite3.connect(DB_PATH)
    conn.execute("INSERT INTO chat_history(role,message,timestamp) VALUES(?,?,?)",
                 (role, message, datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
    conn.commit(); conn.close()

def db_load_chat(limit=100):
    conn = sqlite3.connect(DB_PATH)
    rows = conn.execute("SELECT role,message,timestamp FROM chat_history"
                        " ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
    conn.close()
    return list(reversed(rows))

def db_clear_chat():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("DELETE FROM chat_history"); conn.commit(); conn.close()

# ══════════════════════════════════════════════════════════════════════════════
#  ARTIFACT HELPERS
# ══════════════════════════════════════════════════════════════════════════════
def make_folder(target: str) -> str:
    ts   = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    safe = re.sub(r"[^\w\-.]","_",target)
    path = os.path.join(scans_dir(), f"{safe}_{ts}")
    os.makedirs(path, exist_ok=True)
    return path

def write_file(folder: str, fname: str, content: str):
    try:
        with open(os.path.join(folder,fname),"w",encoding="utf-8") as f:
            f.write(content)
    except: pass

def open_folder(path: str):
    try:
        if platform.system() == "Windows":
            os.startfile(path)
        elif platform.system() == "Darwin":
            subprocess.Popen(["open", path])
        else:
            subprocess.Popen(["xdg-open", path])
    except: pass

# ══════════════════════════════════════════════════════════════════════════════
#  NETWORK UTILITIES
# ══════════════════════════════════════════════════════════════════════════════
def local_ip() -> str:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8",80)); ip = s.getsockname()[0]; s.close(); return ip
    except: return "127.0.0.1"

def local_network() -> str:
    """Return network in CIDR, e.g. 192.168.1.0/24"""
    ip = local_ip()
    parts = ip.split(".")
    return f"{parts[0]}.{parts[1]}.{parts[2]}.0/24"

def resolve(domain: str):
    try: return socket.gethostbyname(domain.strip())
    except: return None

def tcp_open(host, port, timeout=1.2) -> bool:
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(timeout)
            return s.connect_ex((host, port)) == 0
    except: return False

def udp_probe(host, port, timeout=2.0) -> bool:
    UDP_PROBES = {
        53: b"\x00\x01\x01\x00\x00\x01\x00\x00\x00\x00\x00\x00\x03www\x06google\x03com\x00\x00\x01\x00\x01",
        123:b"\x1b"+47*b"\x00",
    }
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.settimeout(timeout)
            s.sendto(UDP_PROBES.get(port,b"\x00"), (host, port))
            data, _ = s.recvfrom(1024)
            return len(data) > 0
    except socket.timeout: return False
    except ConnectionRefusedError: return False
    except: return False

def ping_host(host: str) -> tuple:
    """Returns (alive:bool, latency_ms:str)"""
    try:
        if platform.system() == "Windows":
            cmd = ["ping","-n","1","-w","1000",host]
        else:
            cmd = ["ping","-c","1","-W","1",host]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3)
        alive  = result.returncode == 0
        # extract latency
        m = re.search(r"[Tt]ime[=<](\d+\.?\d*)\s?ms", result.stdout)
        latency = f"{m.group(1)} ms" if m else ("—" if not alive else "< 1 ms")
        return alive, latency
    except: return False, "—"

def get_hostname(ip: str) -> str:
    try: return socket.gethostbyaddr(ip)[0]
    except: return "—"

def get_mac_arp(ip: str) -> str:
    """Try to read MAC from ARP table (Windows/Linux)."""
    try:
        if platform.system() == "Windows":
            out = subprocess.check_output(["arp","-a",ip], text=True, timeout=3)
        else:
            out = subprocess.check_output(["arp","-n",ip], text=True, timeout=3)
        m = re.search(r"([0-9a-fA-F]{2}[:\-][0-9a-fA-F]{2}[:\-][0-9a-fA-F]{2}"
                      r"[:\-][0-9a-fA-F]{2}[:\-][0-9a-fA-F]{2}[:\-][0-9a-fA-F]{2})",
                      out)
        return m.group(1).upper() if m else "—"
    except: return "—"

def grab_banner(host, port) -> str:
    try:
        if port in (80,8000,8080,8888) and REQUESTS_OK:
            r = requests.get(f"http://{host}:{port}",timeout=3,allow_redirects=False)
            srv = r.headers.get("Server",""); pw = r.headers.get("X-Powered-By","")
            return " | ".join(filter(None,[srv,pw])) or f"HTTP {r.status_code}"
        if port in (443,8443) and REQUESTS_OK:
            import urllib3; urllib3.disable_warnings()
            r = requests.get(f"https://{host}:{port}",timeout=3,
                             verify=False,allow_redirects=False)
            return r.headers.get("Server",f"HTTPS {r.status_code}")
        if port==21 and FTP_OK:
            ftp=ftplib.FTP(); ftp.connect(host,21,timeout=3)
            b=ftp.getwelcome(); ftp.quit(); return b
        if port==25 and SMTP_OK:
            sm=smtplib.SMTP(host,25,timeout=3)
            b=sm.ehlo()[1].decode(errors="ignore"); sm.quit()
            return b.split("\n")[0][:100]
        if port==22:
            with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as s:
                s.settimeout(3); s.connect((host,22))
                return s.recv(256).decode(errors="ignore").strip()
        with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as s:
            s.settimeout(2); s.connect((host,port))
            s.sendall(b"HEAD / HTTP/1.0\r\nHost: x\r\n\r\n")
            raw=s.recv(512).decode(errors="ignore").strip()
            return raw.split("\n")[0][:120]
    except: return ""

def svc_name(port, proto="tcp") -> str:
    try: return socket.getservbyport(port, proto)
    except:
        T={20:"ftp-data",21:"ftp",22:"ssh",23:"telnet",25:"smtp",53:"dns",
           67:"dhcp",80:"http",110:"pop3",135:"msrpc",137:"netbios",
           139:"netbios-ssn",143:"imap",443:"https",445:"smb",465:"smtps",
           587:"smtp-sub",993:"imaps",995:"pop3s",1433:"mssql",1521:"oracle",
           3306:"mysql",3389:"rdp",5432:"postgresql",5900:"vnc",5985:"winrm",
           6379:"redis",8000:"http-dev",8080:"http-alt",8443:"https-alt",
           8888:"jupyter",9200:"elasticsearch",27017:"mongodb"}
        return T.get(port,"unknown")

# ══════════════════════════════════════════════════════════════════════════════
#  VULNERABILITY DATABASE
# ══════════════════════════════════════════════════════════════════════════════
VULN_DB = {
    21:[{"severity":"High","cve":"CVE-2010-4221","cvss":"7.5",
         "description":"FTP — unencrypted, anonymous login risk",
         "risk":"Credentials transmitted in cleartext; anonymous access may expose files",
         "mitigation":"Use SFTP/FTPS; disable anonymous login; enforce strong passwords; firewall port 21"}],
    22:[{"severity":"High","cve":"CVE-2023-48795","cvss":"5.9",
         "description":"SSH Terrapin — handshake security degraded",
         "risk":"MITM can downgrade encryption during SSH handshake",
         "mitigation":"Patch OpenSSH ≥9.6; enforce key-based auth; disable root login"}],
    23:[{"severity":"Critical","cve":"N/A","cvss":"9.8",
         "description":"Telnet — cleartext credentials",
         "risk":"All traffic including passwords interceptable",
         "mitigation":"Disable Telnet immediately; replace with SSH 2.0; block port 23"}],
    25:[{"severity":"High","cve":"Multiple","cvss":"7.5",
         "description":"SMTP — open relay and user enumeration",
         "risk":"Spam distribution, phishing, user enumeration",
         "mitigation":"Require SMTP AUTH; configure SPF/DKIM/DMARC; rate-limit"}],
    53:[{"severity":"Medium","cve":"CVE-2023-50868","cvss":"5.3",
         "description":"DNS — amplification and cache poisoning",
         "risk":"DDoS amplification, DNS hijacking",
         "mitigation":"Enable DNSSEC; restrict recursive queries; rate-limit"}],
    80:[{"severity":"High","cve":"Multiple","cvss":"7.5",
         "description":"HTTP — unencrypted web service",
         "risk":"Data interceptable; session hijacking via MITM",
         "mitigation":"Migrate to HTTPS; implement HSTS; redirect HTTP→HTTPS"}],
    110:[{"severity":"High","cve":"N/A","cvss":"7.5",
          "description":"POP3 — cleartext email","risk":"Email credentials readable",
          "mitigation":"Replace with POP3S on port 995"}],
    135:[{"severity":"Critical","cve":"CVE-2023-23397","cvss":"9.8",
          "description":"MS-RPC — remote code execution vector",
          "risk":"Full system compromise via RPC exploitation",
          "mitigation":"Block port 135; apply all Windows patches"}],
    139:[{"severity":"High","cve":"CVE-2021-1675","cvss":"7.8",
          "description":"NetBIOS-SSN — SMB relay, null session",
          "risk":"Lateral movement, credential harvesting",
          "mitigation":"Disable NetBIOS over TCP/IP; block 137-139"}],
    143:[{"severity":"High","cve":"N/A","cvss":"7.5",
          "description":"IMAP — cleartext email","risk":"Login credentials visible",
          "mitigation":"Replace with IMAPS on port 993"}],
    443:[{"severity":"Medium","cve":"Multiple","cvss":"5.9",
          "description":"HTTPS — verify TLS version and cipher config",
          "risk":"MITM if TLS 1.0/1.1 or weak ciphers enabled",
          "mitigation":"Enforce TLS 1.2+; strong cipher suites; HSTS"}],
    445:[{"severity":"Critical","cve":"MS17-010","cvss":"9.8",
          "description":"SMB — EternalBlue, PrintNightmare RCE",
          "risk":"Remote code execution, ransomware deployment",
          "mitigation":"Block SMB externally; apply MS17-010 patch; use SMB 3.1.1+"}],
    1433:[{"severity":"Critical","cve":"Multiple","cvss":"9.8",
           "description":"MSSQL — SQL injection, authentication bypass",
           "risk":"Full database compromise, data exfiltration",
           "mitigation":"Firewall 1433; Windows Auth; disable 'sa' account"}],
    3306:[{"severity":"Critical","cve":"CVE-2021-2122","cvss":"9.1",
           "description":"MySQL exposed — potential root/no-password",
           "risk":"Full database read/write/delete",
           "mitigation":"Bind to 127.0.0.1; require SSL; strong root password"}],
    3389:[{"severity":"Critical","cve":"CVE-2019-0708","cvss":"9.8",
           "description":"RDP — BlueKeep remote code execution",
           "risk":"Complete remote system takeover",
           "mitigation":"VPN for RDP; enable NLA; block 3389 publicly; patch"}],
    5432:[{"severity":"Critical","cve":"Multiple","cvss":"9.1",
           "description":"PostgreSQL — trust auth risk",
           "risk":"Data breach, arbitrary SQL execution",
           "mitigation":"Restrict pg_hba; enforce SSL; firewall"}],
    5900:[{"severity":"Critical","cve":"N/A","cvss":"9.8",
           "description":"VNC — weak/no auth, unencrypted",
           "risk":"Full remote desktop access",
           "mitigation":"Strong password; SSH tunnel; firewall 5900"}],
    5985:[{"severity":"High","cve":"N/A","cvss":"8.1",
           "description":"WinRM — PowerShell remoting exposed",
           "risk":"Remote command execution",
           "mitigation":"Restrict; require HTTPS (5986); firewall"}],
    6379:[{"severity":"Critical","cve":"N/A","cvss":"9.8",
           "description":"Redis — no authentication by default",
           "risk":"Full data access; RCE via config write",
           "mitigation":"Enable requirepass; bind 127.0.0.1; firewall 6379"}],
    8080:[{"severity":"High","cve":"Multiple","cvss":"7.5",
           "description":"HTTP-alt — admin consoles often exposed",
           "risk":"Default credentials, admin panel access",
           "mitigation":"Change default creds; restrict; reverse proxy"}],
    9200:[{"severity":"Critical","cve":"N/A","cvss":"9.8",
           "description":"Elasticsearch — no auth by default",
           "risk":"All data publicly readable",
           "mitigation":"Enable xpack security; firewall; TLS"}],
    27017:[{"severity":"Critical","cve":"N/A","cvss":"9.8",
            "description":"MongoDB — no auth in default config",
            "risk":"Full database read/write/delete",
            "mitigation":"Enable auth; bind localhost; firewall; update"}],
}

NMAP_CMDS = {
    21:["nmap -sV -p 21 {ip}","nmap --script ftp-anon,ftp-bounce,ftp-brute -p 21 {ip}"],
    22:["nmap -sV -p 22 {ip}","nmap --script ssh-brute,ssh2-enum-algos -p 22 {ip}"],
    23:["nmap -sV -p 23 {ip}","nmap --script telnet-brute -p 23 {ip}"],
    25:["nmap -sV -p 25 {ip}","nmap --script smtp-open-relay,smtp-enum-users -p 25 {ip}"],
    53:["nmap -sV -p 53 {ip}","nmap --script dns-zone-transfer,dns-recursion -p 53 {ip}"],
    80:["nmap -sV -p 80 {ip}","nmap --script http-methods,http-sql-injection,http-vuln* -p 80 {ip}"],
    443:["nmap -sV -p 443 {ip}","nmap --script ssl-enum-ciphers,ssl-heartbleed -p 443 {ip}"],
    445:["nmap -sV -p 445 {ip}","nmap --script smb-vuln-ms17-010,smb-enum-shares -p 445 {ip}"],
    3306:["nmap -sV -p 3306 {ip}","nmap --script mysql-brute,mysql-empty-password -p 3306 {ip}"],
    3389:["nmap -sV -p 3389 {ip}","nmap --script rdp-vuln-ms12-020,rdp-enum-encryption -p 3389 {ip}"],
    6379:["nmap -sV -p 6379 {ip}","nmap --script redis-info -p 6379 {ip}"],
    8080:["nmap -sV -p 8080 {ip}","nmap --script http-methods,http-open-proxy -p 8080 {ip}"],
    27017:["nmap -sV -p 27017 {ip}","nmap --script mongodb-info,mongodb-databases -p 27017 {ip}"],
}

MANUAL_CHECKS={
    21:[("Anonymous Login","ftp {ip}  → user: anonymous  pass: (blank)"),
        ("FTP Version","Connect and read welcome banner")],
    22:[("SSH Banner","ssh -v {ip}"),("Key Auth","Verify PasswordAuthentication=no")],
    23:[("Telnet Connect","telnet {ip}  → ALL TRAFFIC CLEARTEXT")],
    25:[("SMTP Banner","telnet {ip} 25  → EHLO test"),
        ("Open Relay","MAIL FROM:<x@x.com>  RCPT TO:<z@z.com>")],
    53:[("Zone Transfer","nslookup -type=AXFR domain.com {ip}"),
        ("Open Resolver","nslookup google.com {ip}")],
    80:[("HTTP Headers","curl -I http://{ip}/"),
        ("Admin Paths","Try /admin /login /wp-admin in browser"),
        ("Default Creds","admin:admin  admin:password")],
    443:[("TLS Version","curl -vI https://{ip}/"),("Cert Check","Check browser padlock")],
    445:[("SMB Null","net use \\\\{ip}\\IPC$"),("EternalBlue","Verify MS17-010 patched")],
    3306:[("MySQL Root","mysql -h {ip} -u root  (blank password)")],
    3389:[("RDP Connect","mstsc /v:{ip}"),("NLA Check","Verify NLA enforced")],
    6379:[("Redis PING","redis-cli -h {ip}  → PING"),("Redis Config","redis-cli -h {ip} CONFIG GET *")],
    8080:[("Admin Console","http://{ip}:8080/manager  /admin"),
          ("Default Creds","tomcat:tomcat  admin:admin")],
}
DEFAULT_CHECKS=[("Version Detect","Connect and read banner from {ip}:{port}"),
                ("Default Creds","Try common username/password combinations")]

KNOCK2_SYSTEM="""You are Knock-2 AI, a cybersecurity assistant by Supraja Technologies Cyber Security Cell.

STRICT RULES:
- ONLY answer cybersecurity questions: port scanning, vulnerabilities, CVEs, network security,
  penetration testing, nmap, firewalls, hardening, OSINT, ethical hacking, malware, CTF, security tools.
- For ANY non-security question, politely refuse and redirect to security topics.
- Never help with: general coding, homework, cooking, entertainment, relationships, etc.
- Always recommend ethical and legal security practices only.
- Sign all responses: — Knock-2 AI | Supraja Technologies Cyber Security Cell

Be concise, technical, and actionable."""

# ══════════════════════════════════════════════════════════════════════════════
#  GEMINI HELPER
# ══════════════════════════════════════════════════════════════════════════════
def gemini_ask(prompt: str, system: str="") -> str:
    api_key = get_api_key()
    if not GEMINI_OK:
        return "❌ google-generativeai not installed.\nRun: pip install google-generativeai"
    if not api_key:
        return ("❌ Gemini API key not set.\n\n"
                "Go to Settings tab to enter your API key.\n"
                "Get a free key at: https://makersuite.google.com/app/apikey")
    import time, random
    models=["gemini-2.5-flash","gemini-2.0-flash","gemini-1.5-flash","gemini-1.5-pro"]
    for attempt in range(4):
        try:
            genai.configure(api_key=api_key)
            model = None
            for m in models:
                try: model=genai.GenerativeModel(m); break
                except: continue
            if model is None: return "❌ No Gemini model available."
            full = f"{system}\n\n{prompt}" if system else prompt
            resp = model.generate_content(full)
            return resp.text or "No response from Gemini."
        except Exception as exc:
            es = str(exc).lower()
            if ("429" in es or "quota" in es) and attempt<3:
                time.sleep(min(2**(attempt+1)+random.uniform(0,1),45))
            else:
                return f"❌ Gemini error: {exc}"
    return "❌ Gemini quota exceeded."

# ══════════════════════════════════════════════════════════════════════════════
#  SCROLLABLE FRAME (fixes all scrolling issues)
# ══════════════════════════════════════════════════════════════════════════════
class ScrollFrame(tk.Frame):
    """A frame with a working vertical scrollbar and mousewheel support."""
    def __init__(self, parent, bg=BG, **kw):
        super().__init__(parent, bg=bg, **kw)
        self._canvas = tk.Canvas(self, bg=bg, bd=0, highlightthickness=0)
        self._vsb    = ttk.Scrollbar(self, orient="vertical",
                                      command=self._canvas.yview)
        self._canvas.configure(yscrollcommand=self._vsb.set)
        self._vsb.pack(side="right", fill="y")
        self._canvas.pack(side="left", fill="both", expand=True)
        self.inner = tk.Frame(self._canvas, bg=bg)
        self._win_id = self._canvas.create_window((0,0), window=self.inner,
                                                   anchor="nw")
        self.inner.bind("<Configure>", self._on_inner_configure)
        self._canvas.bind("<Configure>", self._on_canvas_configure)
        self._bind_mousewheel(self)

    def _on_inner_configure(self, _e):
        self._canvas.configure(scrollregion=self._canvas.bbox("all"))

    def _on_canvas_configure(self, e):
        self._canvas.itemconfig(self._win_id, width=e.width)

    def _bind_mousewheel(self, widget):
        widget.bind("<MouseWheel>",   self._on_scroll_win, add="+")
        widget.bind("<Button-4>",     self._on_scroll_lin, add="+")
        widget.bind("<Button-5>",     self._on_scroll_lin, add="+")
        for child in widget.winfo_children():
            self._bind_mousewheel(child)

    def _on_scroll_win(self, event):
        self._canvas.yview_scroll(int(-1*(event.delta/120)), "units")

    def _on_scroll_lin(self, event):
        direction = -1 if event.num==4 else 1
        self._canvas.yview_scroll(direction, "units")

    def add_child(self, widget):
        """Re-bind mousewheel after adding new children."""
        self._bind_mousewheel(widget)

# ══════════════════════════════════════════════════════════════════════════════
#  TOOLTIP
# ══════════════════════════════════════════════════════════════════════════════
class ToolTip:
    def __init__(self, widget, text):
        self._tip = None
        widget.bind("<Enter>", lambda _: self._show(widget, text))
        widget.bind("<Leave>", lambda _: self._hide())
    def _show(self, w, text):
        x=w.winfo_rootx()+20; y=w.winfo_rooty()+28
        self._tip=tk.Toplevel(w); self._tip.wm_overrideredirect(True)
        self._tip.wm_geometry(f"+{x}+{y}")
        tk.Label(self._tip,text=text,bg=CARD_BG,fg=TEXT,relief="solid",
                 borderwidth=1,padx=8,pady=4,font=("Segoe UI",9)).pack()
    def _hide(self):
        if self._tip: self._tip.destroy(); self._tip=None

# ══════════════════════════════════════════════════════════════════════════════
#  KNOCK-2 AI CHATBOT WINDOW  (with persistent history)
# ══════════════════════════════════════════════════════════════════════════════
class Knock2Window:
    def __init__(self, parent):
        self.parent = parent
        self.win = tk.Toplevel(parent)
        self.win.title("Knock-2 AI — Cybersecurity Assistant")
        self.win.geometry("500x700"); self.win.configure(bg=KNOCK_BG)
        self._build(); self._load_history()

    def _build(self):
        hdr=tk.Frame(self.win,bg="#0f1824",pady=10); hdr.pack(fill="x")
        tk.Label(hdr,text="🔐 Knock-2 AI",font=("Segoe UI",14,"bold"),
                 fg=KNOCK_ACC,bg="#0f1824").pack(side="left",padx=14)
        tk.Label(hdr,text="Cybersecurity Only",font=("Segoe UI",9),
                 fg=TEXT_DIM,bg="#0f1824").pack(side="left")
        tk.Label(hdr,text="Supraja Technologies",font=("Segoe UI",8),
                 fg=ACCENT,bg="#0f1824").pack(side="right",padx=14)
        # clear button
        tk.Button(hdr,text="🗑 Clear",bg="#0f1824",fg=RED,font=("Segoe UI",8),
                  bd=0,cursor="hand2",command=self._clear_hist).pack(
                      side="right",padx=4)
        tk.Frame(self.win,bg=BORDER,height=1).pack(fill="x")

        self.chat=scrolledtext.ScrolledText(
            self.win,bg="#080e18",fg=TEXT,font=("Segoe UI",9),
            bd=0,padx=12,pady=10,insertbackground=KNOCK_ACC,
            wrap="word",state="disabled")
        self.chat.pack(fill="both",expand=True)
        self.chat.tag_configure("user",foreground=KNOCK_ACC,
                                font=("Segoe UI",9,"bold"))
        self.chat.tag_configure("ai",  foreground=TEXT)
        self.chat.tag_configure("sys", foreground=TEXT_DIM,
                                font=("Segoe UI",8,"italic"))
        self.chat.tag_configure("ts",  foreground=BORDER,
                                font=("Segoe UI",7))

        # quick prompts
        qf=tk.Frame(self.win,bg=KNOCK_BG); qf.pack(fill="x",padx=8,pady=(4,0))
        tk.Label(qf,text="Quick:",fg=TEXT_DIM,bg=KNOCK_BG,
                 font=("Segoe UI",8)).pack(side="left")
        for q in ["nmap for SMB","fix RDP","Heartbleed","harden SSH",
                  "Redis CVE","OWASP Top 10"]:
            tk.Button(qf,text=q,bg=CARD_BG,fg=KNOCK_ACC,
                      font=("Segoe UI",8),bd=0,padx=5,pady=2,cursor="hand2",
                      command=lambda t=q: self._send(t)).pack(
                          side="left",padx=2,pady=3)

        # input
        inf=tk.Frame(self.win,bg="#0f1824",pady=8); inf.pack(fill="x")
        self.inp=tk.Entry(inf,bg=CARD_BG,fg=TEXT,font=("Segoe UI",10),
                           insertbackground=KNOCK_ACC,relief="flat",
                           highlightthickness=1,highlightbackground=BORDER,
                           highlightcolor=KNOCK_ACC)
        self.inp.pack(side="left",fill="x",expand=True,padx=(12,6))
        self.inp.bind("<Return>",lambda _: self._send())
        tk.Button(inf,text="Ask →",bg=KNOCK_ACC,fg="#000",
                  font=("Segoe UI",10,"bold"),bd=0,padx=12,pady=5,
                  cursor="hand2",command=self._send).pack(side="right",padx=(0,12))

    def _load_history(self):
        rows = db_load_chat(60)
        if rows:
            self._append("sys","— Previous conversation —\n")
            for role, msg, ts in rows:
                tag = "user" if role=="user" else "ai"
                prefix = "You" if role=="user" else "Knock-2"
                self._append(tag, f"{prefix}: {msg}")
                self._append("ts", f"  {ts}\n")
        else:
            self._append("sys","Knock-2 AI ready. Ask me anything about cybersecurity.\n")

    def _append(self, tag: str, text: str):
        self.chat.configure(state="normal")
        self.chat.insert("end", text+"\n", tag)
        self.chat.see("end"); self.chat.configure(state="disabled")

    def _send(self, text=None):
        msg = text or self.inp.get().strip()
        if not msg: return
        self.inp.delete(0,"end")
        self._append("user", f"You: {msg}")
        db_save_chat("user", msg)
        self._append("sys","Knock-2 thinking…")
        threading.Thread(target=self._ask, args=(msg,), daemon=True).start()

    def _ask(self, msg: str):
        resp = gemini_ask(msg, KNOCK2_SYSTEM)
        db_save_chat("ai", resp)
        self.win.after(0, lambda: self._append("ai", f"Knock-2: {resp}\n"))

    def _clear_hist(self):
        if messagebox.askyesno("Clear", "Clear all chat history?", parent=self.win):
            db_clear_chat()
            self.chat.configure(state="normal")
            self.chat.delete("1.0","end")
            self.chat.configure(state="disabled")
            self._append("sys","Chat history cleared.\n")

# ══════════════════════════════════════════════════════════════════════════════
#  MAIN APPLICATION
# ══════════════════════════════════════════════════════════════════════════════
class AdvancedPortScanner:
    def __init__(self, root: tk.Tk):
        self.root = root
        self.root.title("Advanced Port Scanner")     # ← clean title
        self.root.geometry("1340x900")
        self.root.minsize(1000,680)
        self.root.configure(bg=BG)
        self._set_icon()

        self.scanning     = False
        self.scan_results = []
        self.vuln_results = {}
        self.nmap_cmds    = {}
        self.ai_text      = ""
        self.scan_folder  = ""
        self.log_lines    = []
        self.err_lines    = []
        self._open_ct     = 0
        self._err_ct      = 0
        self.assessor_var = tk.StringVar(value=CFG.get("assessor_name","Security Analyst"))

        init_db()
        self._styles()
        self._build_ui()

    def _set_icon(self):
        """Draw a simple lock icon as the window icon (no external file needed)."""
        try:
            img = tk.PhotoImage(width=32, height=32)
            # draw a simple green lock shape pixel by pixel
            for x in range(8,24):
                for y in range(14,28):
                    img.put("#00ff88",(x,y))
            for x in range(10,22):
                for y in range(6,16):
                    img.put("#00ff88",(x,y))
            for x in range(13,19):
                for y in range(4,14):
                    img.put(BG,(x,y))
            self.root.iconphoto(True, img)
        except: pass

    def _styles(self):
        s=ttk.Style(); s.theme_use("default")
        for n in("Scanner","History","Vuln","Net"):
            s.configure(f"{n}.Treeview",background=CARD_BG,foreground=TEXT,
                        rowheight=26,fieldbackground=CARD_BG,borderwidth=0,
                        font=("Segoe UI",9))
            s.configure(f"{n}.Treeview.Heading",background=SIDEBAR_BG,
                        foreground=ACCENT,font=("Segoe UI",9,"bold"),relief="flat")
            s.map(f"{n}.Treeview",
                  background=[("selected",ACCENT2)],foreground=[("selected","#000")])
        s.configure("Horizontal.TProgressbar",troughcolor=CARD_BG,background=ACCENT)
        s.configure("TCombobox",fieldbackground=CARD_BG,background=CARD_BG,
                    foreground=TEXT,arrowcolor=TEXT)
        s.map("TCombobox",fieldbackground=[("readonly",CARD_BG)])

    def _build_ui(self):
        # ── Sidebar
        sb=tk.Frame(self.root,bg=SIDEBAR_BG,width=185)
        sb.pack(side="left",fill="y"); sb.pack_propagate(False)

        # Logo area
        lf=tk.Frame(sb,bg=SIDEBAR_BG,pady=14); lf.pack(fill="x")
        tk.Label(lf,text="⚡",font=("Segoe UI",22),fg=ACCENT,bg=SIDEBAR_BG).pack()
        tk.Label(lf,text="Advanced Port\nScanner",font=("Segoe UI",9,"bold"),
                 fg=ACCENT,bg=SIDEBAR_BG,justify="center").pack()
        tk.Label(lf,text="v5.0  Supraja Tech",font=("Segoe UI",7),
                 fg=TEXT_DIM,bg=SIDEBAR_BG).pack()

        tk.Frame(sb,bg=BORDER,height=1).pack(fill="x",padx=12)

        self._nav={}
        for key,icon,lbl in [("scanner","🔍"," Scanner"),
                              ("network","📡"," My Network"),
                              ("history","📋"," History"),
                              ("tools",  "🛠"," Tools"),
                              ("settings","⚙️"," Settings"),
                              ("about",  "🏢"," About")]:
            b=tk.Button(sb,text=f"  {icon}{lbl}",anchor="w",
                        bg=SIDEBAR_BG,fg=TEXT,bd=0,padx=18,pady=11,
                        font=("Segoe UI",10),cursor="hand2",
                        activebackground=CARD_BG,activeforeground=ACCENT,
                        command=lambda k=key:self._show(k))
            b.pack(fill="x"); self._nav[key]=b

        tk.Frame(sb,bg=BORDER,height=1).pack(fill="x",padx=12,pady=4)
        tk.Button(sb,text="🔐 Knock-2 AI\nCyber Assistant",
                  bg="#0f1824",fg=KNOCK_ACC,font=("Segoe UI",9,"bold"),
                  bd=0,padx=10,pady=10,cursor="hand2",justify="center",
                  command=lambda:Knock2Window(self.root)).pack(
                      fill="x",padx=8,pady=4)

        tk.Frame(sb,bg=BORDER,height=1).pack(fill="x",padx=12,side="bottom",pady=4)
        tk.Label(sb,text=f"Local IP\n{local_ip()}",font=("Segoe UI",8),
                 fg=TEXT_DIM,bg=SIDEBAR_BG,pady=8).pack(side="bottom")

        self.content=tk.Frame(self.root,bg=BG)
        self.content.pack(side="right",fill="both",expand=True)
        self._pages={
            "scanner": self._page_scanner(),
            "network": self._page_network(),
            "history": self._page_history(),
            "tools":   self._page_tools(),
            "settings":self._page_settings(),
            "about":   self._page_about(),
        }
        self._show("scanner")

    def _show(self, page: str):
        for f in self._pages.values(): f.pack_forget()
        self._pages[page].pack(fill="both",expand=True)
        for k,b in self._nav.items():
            b.configure(bg=CARD_BG if k==page else SIDEBAR_BG,
                        fg=ACCENT  if k==page else TEXT)

    # ─────────────────────────────────────────────────────────────────────
    #  SHARED ENTRY WIDGET
    # ─────────────────────────────────────────────────────────────────────
    def _ent(self, parent, default="", var=None):
        kw = {"textvariable": var} if var else {}
        e  = tk.Entry(parent,bg="#0d1117",fg=TEXT,insertbackground=ACCENT,
                      font=("Segoe UI",10),relief="flat",highlightthickness=1,
                      highlightbackground=BORDER,highlightcolor=ACCENT,**kw)
        if not var: e.insert(0,default)
        return e

    # ══════════════════════════════════════════════════════════════════════
    #  SCANNER PAGE
    # ══════════════════════════════════════════════════════════════════════
    def _page_scanner(self):
        frame=tk.Frame(self.content,bg=BG)

        hdr=tk.Frame(frame,bg=BG); hdr.pack(fill="x",padx=20,pady=(12,4))
        tk.Label(hdr,text="Port Scanner",font=("Segoe UI",17,"bold"),
                 fg=ACCENT,bg=BG).pack(side="left")
        tk.Label(hdr,text="Enterprise v5.0  |  Supraja Technologies",
                 font=("Segoe UI",9),fg=TEXT_DIM,bg=BG).pack(side="left",padx=12)

        # prompt bar
        pf=tk.Frame(frame,bg=CARD_BG); pf.pack(fill="x",padx=20,pady=(0,6))
        pi=tk.Frame(pf,bg=CARD_BG,padx=12,pady=8); pi.pack(fill="x")
        tk.Label(pi,text="💬 Prompt:",fg=KNOCK_ACC,bg=CARD_BG,
                 font=("Segoe UI",9,"bold")).pack(side="left")
        self.prompt_e=tk.Entry(pi,bg="#0d1117",fg=TEXT,font=("Segoe UI",9),
                                insertbackground=ACCENT,relief="flat",
                                highlightthickness=1,highlightbackground=BORDER,
                                highlightcolor=KNOCK_ACC)
        self.prompt_e.insert(0,'e.g. "scan 192.168.1.1 ports 80-443"')
        self.prompt_e.bind("<FocusIn>",
            lambda _: self.prompt_e.delete(0,"end")
            if "e.g." in self.prompt_e.get() else None)
        self.prompt_e.pack(side="left",fill="x",expand=True,padx=8)
        tk.Button(pi,text="🤖 AI Parse",bg=KNOCK_ACC,fg="#000",
                  font=("Segoe UI",9,"bold"),bd=0,padx=10,pady=3,
                  cursor="hand2",command=self._ai_parse).pack(side="left",padx=(0,4))
        tk.Button(pi,text="Parse",bg=CARD_BG,fg=TEXT_DIM,
                  font=("Segoe UI",9),bd=1,relief="solid",padx=8,pady=3,
                  cursor="hand2",command=self._parse_prompt).pack(side="left")

        # input card
        card=tk.Frame(frame,bg=CARD_BG); card.pack(fill="x",padx=20,pady=(0,6))
        inner=tk.Frame(card,bg=CARD_BG,padx=20,pady=12); inner.pack(fill="x")

        r1=tk.Frame(inner,bg=CARD_BG); r1.pack(fill="x",pady=3)
        tk.Label(r1,text="Domain:",fg=TEXT_DIM,bg=CARD_BG,
                 font=("Segoe UI",9),width=10,anchor="w").grid(row=0,column=0)
        self.domain_e=self._ent(r1); self.domain_e.grid(row=0,column=1,sticky="ew",padx=(8,8))
        rb=tk.Button(r1,text="Resolve →",bg=ACCENT2,fg="#000",
                     font=("Segoe UI",9,"bold"),bd=0,padx=9,pady=3,
                     cursor="hand2",command=self._resolve)
        rb.grid(row=0,column=2,padx=(0,20)); ToolTip(rb,"Domain → IP")
        tk.Label(r1,text="IP:",fg=TEXT_DIM,bg=CARD_BG,
                 font=("Segoe UI",9),width=4,anchor="w").grid(row=0,column=3)
        self.ip_e=self._ent(r1); self.ip_e.grid(row=0,column=4,sticky="ew",padx=(8,0))
        r1.columnconfigure(1,weight=2); r1.columnconfigure(4,weight=2)

        r2=tk.Frame(inner,bg=CARD_BG); r2.pack(fill="x",pady=3)
        tk.Label(r2,text="Start Port:",fg=TEXT_DIM,bg=CARD_BG,
                 font=("Segoe UI",9),width=10,anchor="w").grid(row=0,column=0)
        self.sp_e=self._ent(r2,"1"); self.sp_e.grid(row=0,column=1,sticky="ew",padx=(8,8))
        tk.Label(r2,text="End Port:",fg=TEXT_DIM,bg=CARD_BG,
                 font=("Segoe UI",9),width=8,anchor="w").grid(row=0,column=2)
        self.ep_e=self._ent(r2,"1024"); self.ep_e.grid(row=0,column=3,sticky="ew",padx=(8,20))
        tk.Label(r2,text="Assessor:",fg=TEXT_DIM,bg=CARD_BG,
                 font=("Segoe UI",9),width=8,anchor="w").grid(row=0,column=4)
        ae=self._ent(r2,"",self.assessor_var)
        ae.grid(row=0,column=5,sticky="ew",padx=(8,0))
        for i in (1,3,5): r2.columnconfigure(i,weight=1)

        r3=tk.Frame(inner,bg=CARD_BG); r3.pack(fill="x",pady=6)
        tk.Label(r3,text="Protocol:",fg=TEXT_DIM,bg=CARD_BG,
                 font=("Segoe UI",9)).pack(side="left")
        self.proto_var=tk.StringVar(value=CFG.get("default_proto","TCP"))
        ttk.Combobox(r3,textvariable=self.proto_var,
                     values=["TCP","UDP","BOTH"],width=7,
                     state="readonly").pack(side="left",padx=(5,14))
        tk.Label(r3,text="Mode:",fg=TEXT_DIM,bg=CARD_BG,
                 font=("Segoe UI",9)).pack(side="left")
        self.mode_var=tk.StringVar(value=CFG.get("default_mode","Traditional"))
        ttk.Combobox(r3,textvariable=self.mode_var,
                     values=["Traditional","Automated"],width=12,
                     state="readonly").pack(side="left",padx=(5,14))
        self.allports_var=tk.BooleanVar(value=False)
        tk.Checkbutton(r3,text="Scan All Ports (0–65535)",
                       variable=self.allports_var,
                       bg=CARD_BG,fg=TEXT,selectcolor=CARD_BG,
                       activebackground=CARD_BG,activeforeground=ACCENT,
                       font=("Segoe UI",9),cursor="hand2").pack(side="left")
        self.stop_btn=tk.Button(r3,text="■ Stop",bg=RED,fg="white",
                                font=("Segoe UI",10,"bold"),bd=0,padx=14,pady=5,
                                cursor="hand2",state="disabled",command=self._stop)
        self.stop_btn.pack(side="right",padx=(6,0))
        self.scan_btn=tk.Button(r3,text="▶ Start Scan",bg=ACCENT,fg="#000",
                                font=("Segoe UI",10,"bold"),bd=0,padx=18,pady=5,
                                cursor="hand2",command=self._start)
        self.scan_btn.pack(side="right")
        tk.Button(r3,text="⬇ CSV",bg=CARD_BG,fg=TEXT_DIM,
                  font=("Segoe UI",9),bd=1,relief="solid",padx=9,pady=4,
                  cursor="hand2",command=self._export_csv).pack(side="right",padx=(0,8))

        # status strip
        ss=tk.Frame(frame,bg=SIDEBAR_BG); ss.pack(fill="x",padx=20,pady=(0,3))
        self.status_var  =tk.StringVar(value="Ready — enter target and click ▶ Start Scan")
        self.curport_var =tk.StringVar(value="Port: —")
        self.openct_var  =tk.StringVar(value="Open: 0")
        self.errct_var   =tk.StringVar(value="Errors: 0")
        tk.Label(ss,textvariable=self.status_var,fg=TEXT_DIM,bg=SIDEBAR_BG,
                 font=("Segoe UI",9),anchor="w").pack(
                     side="left",fill="x",expand=True,padx=8,pady=4)
        for var,fg in[(self.openct_var,GREEN),(self.errct_var,RED),(self.curport_var,ACCENT2)]:
            tk.Label(ss,textvariable=var,fg=fg,bg=SIDEBAR_BG,
                     font=("Segoe UI",9)).pack(side="right",padx=8)

        self.prog_var=tk.DoubleVar()
        ttk.Progressbar(frame,variable=self.prog_var,maximum=100,
                        style="Horizontal.TProgressbar").pack(
                            fill="x",padx=20,pady=(0,4))

        # results table
        tf=tk.Frame(frame,bg=BG); tf.pack(fill="both",expand=True,padx=20,pady=(0,5))
        cols=("Port","Service","Protocol","Status","Banner / Version")
        self.tree=ttk.Treeview(tf,columns=cols,show="headings",style="Scanner.Treeview")
        for col,w in zip(cols,[72,110,80,68,0]):
            self.tree.heading(col,text=col,command=lambda c=col:self._sort(c))
            self.tree.column(col,width=w,stretch=(col=="Banner / Version"))
        self.tree.tag_configure("open",foreground=GREEN)
        vsb=ttk.Scrollbar(tf,orient="vertical",  command=self.tree.yview)
        hsb=ttk.Scrollbar(tf,orient="horizontal",command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set,xscrollcommand=hsb.set)
        self.tree.grid(row=0,column=0,sticky="nsew")
        vsb.grid(row=0,column=1,sticky="ns")
        hsb.grid(row=1,column=0,sticky="ew")
        tf.rowconfigure(0,weight=1); tf.columnconfigure(0,weight=1)
        self.tree.bind("<Double-1>",self._on_dbl)
        ToolTip(self.tree,"Double-click for deep analysis & Nmap AI commands")

        self.empty_lbl=tk.Label(frame,text="No open ports found on target.",
                                fg=TEXT_DIM,bg=BG,font=("Segoe UI",11))

        # vuln summary
        vs=tk.Frame(frame,bg=CARD_BG); vs.pack(fill="x",padx=20,pady=(0,3))
        tk.Label(vs,text="🔴 Vuln:",fg=RED,bg=CARD_BG,
                 font=("Segoe UI",9,"bold"),padx=10,pady=5).pack(side="left")
        self.vuln_sum_var=tk.StringVar(value="Run a scan.")
        tk.Label(vs,textvariable=self.vuln_sum_var,fg=TEXT,bg=CARD_BG,
                 font=("Segoe UI",9)).pack(side="left")
        tk.Button(vs,text="View All Vulns",bg=CARD_BG,fg=ORANGE,
                  font=("Segoe UI",9),bd=1,relief="solid",padx=8,pady=3,
                  cursor="hand2",command=self._show_vulns).pack(side="right",padx=10,pady=4)

        # AI panel
        aih=tk.Frame(frame,bg=CARD_BG); aih.pack(fill="x",padx=20)
        tk.Label(aih,text="🤖 AI Report (Google Gemini)",
                 fg=ACCENT,bg=CARD_BG,font=("Segoe UI",10,"bold"),
                 padx=12,pady=6).pack(side="left")
        for txt,cmd,bgc,fgc in [
            ("Save PDF",       self._save_pdf,   CARD_BG,TEXT_DIM),
            ("Mitigation PDF", self._save_mit,   CARD_BG,TEXT_DIM),
            ("Generate Report",self._ai_report,  ACCENT2,"#000"),
        ]:
            tk.Button(aih,text=txt,bg=bgc,fg=fgc,
                      font=("Segoe UI",9,"bold" if bgc==ACCENT2 else "normal"),
                      bd=0 if bgc==ACCENT2 else 1,
                      relief="flat" if bgc==ACCENT2 else "solid",
                      padx=10,pady=3,cursor="hand2",command=cmd).pack(
                          side="right",padx=(4,0) if txt!="Generate Report" else (10,0),pady=5)

        self.ai_out=scrolledtext.ScrolledText(
            frame,height=6,bg=SIDEBAR_BG,fg=TEXT,font=("Consolas",9),
            insertbackground=ACCENT,bd=0,padx=12,pady=8)
        self.ai_out.pack(fill="x",padx=20,pady=(0,12))
        self._ai_write("AI report appears here after scanning.\n"
                       "Set Gemini API key in ⚙️ Settings tab.  "
                       "Automated mode = fully hands-free.")
        return frame

    def _ai_write(self, text: str):
        self.ai_out.configure(state="normal")
        self.ai_out.delete("1.0","end")
        self.ai_out.insert("end",text)
        self.ai_out.configure(state="disabled")

    def _sort(self, col):
        items=[(self.tree.set(i,col),i) for i in self.tree.get_children()]
        try: items.sort(key=lambda x:int(x[0]))
        except: items.sort()
        for idx,(_,i) in enumerate(items): self.tree.move(i,"",idx)

    # ══════════════════════════════════════════════════════════════════════
    #  NETWORK SCANNER PAGE
    # ══════════════════════════════════════════════════════════════════════
    def _page_network(self):
        frame=tk.Frame(self.content,bg=BG)
        tk.Label(frame,text="📡 My Network Scanner",
                 font=("Segoe UI",17,"bold"),fg=ACCENT,bg=BG,pady=12).pack()

        # Controls
        cf=tk.Frame(frame,bg=CARD_BG); cf.pack(fill="x",padx=20,pady=(0,8))
        ci=tk.Frame(cf,bg=CARD_BG,padx=16,pady=12); ci.pack(fill="x")
        tk.Label(ci,text="Network CIDR:",fg=TEXT_DIM,bg=CARD_BG,
                 font=("Segoe UI",9)).pack(side="left")
        self.net_range_var=tk.StringVar(value=local_network())
        ne=tk.Entry(ci,textvariable=self.net_range_var,bg="#0d1117",fg=TEXT,
                    font=("Segoe UI",10),relief="flat",highlightthickness=1,
                    highlightbackground=BORDER,highlightcolor=ACCENT,width=20)
        ne.pack(side="left",padx=(8,14))
        self.net_status_var=tk.StringVar(value="Ready")
        tk.Label(ci,textvariable=self.net_status_var,fg=TEXT_DIM,bg=CARD_BG,
                 font=("Segoe UI",9)).pack(side="left",expand=True,fill="x")
        self.net_stop_flag=[False]
        self.net_stop_btn=tk.Button(ci,text="■ Stop",bg=RED,fg="white",
                                     font=("Segoe UI",9,"bold"),bd=0,padx=10,pady=4,
                                     cursor="hand2",state="disabled",
                                     command=lambda:self.net_stop_flag.__setitem__(0,True))
        self.net_stop_btn.pack(side="right",padx=(6,0))
        tk.Button(ci,text="🔍 Scan Network",bg=ACCENT,fg="#000",
                  font=("Segoe UI",10,"bold"),bd=0,padx=16,pady=5,
                  cursor="hand2",command=self._scan_network).pack(side="right")

        self.net_prog=tk.DoubleVar()
        ttk.Progressbar(frame,variable=self.net_prog,maximum=100,
                        style="Horizontal.TProgressbar").pack(
                            fill="x",padx=20,pady=(0,6))

        # Device table
        tf=tk.Frame(frame,bg=BG); tf.pack(fill="both",expand=True,padx=20,pady=(0,6))
        cols=("IP","Hostname","MAC","Latency","Status","Open Ports")
        self.net_tree=ttk.Treeview(tf,columns=cols,show="headings",style="Net.Treeview")
        widths=[130,200,155,80,70,0]
        for col,w in zip(cols,widths):
            self.net_tree.heading(col,text=col)
            self.net_tree.column(col,width=w,stretch=(col=="Open Ports"))
        self.net_tree.tag_configure("alive",foreground=GREEN)
        self.net_tree.tag_configure("dead", foreground=TEXT_DIM)
        vsb_n=ttk.Scrollbar(tf,orient="vertical",  command=self.net_tree.yview)
        hsb_n=ttk.Scrollbar(tf,orient="horizontal",command=self.net_tree.xview)
        self.net_tree.configure(yscrollcommand=vsb_n.set,xscrollcommand=hsb_n.set)
        self.net_tree.grid(row=0,column=0,sticky="nsew")
        vsb_n.grid(row=0,column=1,sticky="ns"); hsb_n.grid(row=1,column=0,sticky="ew")
        tf.rowconfigure(0,weight=1); tf.columnconfigure(0,weight=1)
        self.net_tree.bind("<Double-1>",self._on_net_dbl)
        ToolTip(self.net_tree,"Double-click a device for interactive network commands")

        # Interactive commands panel
        cp=tk.Frame(frame,bg=CARD_BG); cp.pack(fill="x",padx=20,pady=(0,4))
        ch=tk.Frame(cp,bg=CARD_BG,padx=14,pady=8); ch.pack(fill="x")
        tk.Label(ch,text="⚡ Interactive Commands:",fg=ACCENT2,bg=CARD_BG,
                 font=("Segoe UI",9,"bold")).pack(side="left")
        self.net_cmd_ip=tk.Entry(ch,bg="#0d1117",fg=TEXT,
                                  font=("Segoe UI",9),width=18,relief="flat",
                                  highlightthickness=1,
                                  highlightbackground=BORDER,highlightcolor=ACCENT)
        self.net_cmd_ip.insert(0,"192.168.1.1"); self.net_cmd_ip.pack(side="left",padx=8)
        for txt,cmd in [("Ping","ping"),("Traceroute","trace"),
                        ("Port Check","portcheck"),("Nslookup","nslookup"),
                        ("ARP","arp"),("Netstat","netstat")]:
            tk.Button(ch,text=txt,bg=CARD_BG,fg=ACCENT2,
                      font=("Segoe UI",9),bd=1,relief="solid",padx=8,pady=3,
                      cursor="hand2",
                      command=lambda c=cmd:self._run_net_cmd(c)).pack(
                          side="left",padx=3)
        tk.Button(ch,text="🤖 Ask AI",bg=KNOCK_ACC,fg="#000",
                  font=("Segoe UI",9,"bold"),bd=0,padx=8,pady=3,cursor="hand2",
                  command=self._net_ai_advice).pack(side="right",padx=4)

        self.net_out=scrolledtext.ScrolledText(
            frame,height=8,bg="#080e18",fg=GREEN,font=("Consolas",9),
            bd=0,padx=10,pady=8,insertbackground=GREEN)
        self.net_out.pack(fill="x",padx=20,pady=(0,12))
        self._net_write("Network scanner ready.\n"
                        "Click '🔍 Scan Network' to discover all devices on your LAN.\n"
                        "Double-click any device for per-device commands.")
        return frame

    def _net_write(self, text: str, append=False):
        self.net_out.configure(state="normal")
        if not append: self.net_out.delete("1.0","end")
        self.net_out.insert("end", text)
        self.net_out.see("end"); self.net_out.configure(state="disabled")

    def _scan_network(self):
        cidr = self.net_range_var.get().strip()
        try: hosts=list(ipaddress.IPv4Network(cidr,strict=False).hosts())
        except:
            messagebox.showerror("Error",f"Invalid CIDR: {cidr}"); return
        self.net_tree.delete(*self.net_tree.get_children())
        self.net_prog.set(0); self.net_stop_flag[0]=False
        self.net_stop_btn.config(state="normal")
        self._net_write(f"Scanning {len(hosts)} hosts on {cidr}…\n",append=False)
        threading.Thread(target=self._run_net_scan,args=(hosts,),daemon=True).start()

    def _run_net_scan(self, hosts):
        total=len(hosts); done=0
        QUICK_PORTS=[22,23,80,139,443,445,3389,8080]
        def scan_one(ip_str):
            alive,lat = ping_host(ip_str)
            if alive:
                hostname = get_hostname(ip_str)
                mac      = get_mac_arp(ip_str)
                open_p   = [str(p) for p in QUICK_PORTS if tcp_open(ip_str,p,0.5)]
                return (ip_str,hostname,mac,lat,"alive",
                        ",".join(open_p) if open_p else "—")
            return None

        with concurrent.futures.ThreadPoolExecutor(max_workers=128) as ex:
            futs={ex.submit(scan_one,str(h)):str(h) for h in hosts}
            for fut in concurrent.futures.as_completed(futs):
                if self.net_stop_flag[0]: break
                done+=1
                pct=100*done/total
                self.root.after(0,lambda v=pct:self.net_prog.set(v))
                self.root.after(0,lambda p=done,t=total:self.net_status_var.set(
                    f"Scanning {p}/{t}…"))
                try:
                    r=fut.result()
                    if r:
                        self.root.after(0,lambda row=r:(
                            self.net_tree.insert("","end",values=row,tags=("alive",)),
                            self._net_write(f"  Found: {row[0]}  {row[1]}  "
                                            f"ports:{row[5]}\n",append=True)
                        ))
                except: pass

        self.root.after(0,lambda:self.net_status_var.set(
            f"Scan done — {self.net_tree.get_children().__len__()} devices found"))
        self.root.after(0,lambda:self.net_stop_btn.config(state="disabled"))
        self.root.after(0,lambda:self.net_prog.set(100))

    def _on_net_dbl(self, _e):
        sel=self.net_tree.selection()
        if not sel: return
        ip=self.net_tree.item(sel[0],"values")[0]
        self.net_cmd_ip.delete(0,"end")
        self.net_cmd_ip.insert(0,ip)
        self._net_write(f"\n--- Selected device: {ip} ---\n",append=True)

    def _run_net_cmd(self, cmd: str):
        ip=self.net_cmd_ip.get().strip()
        if not ip: return
        self._net_write(f"\n$ Running {cmd} on {ip}…\n",append=True)
        def do():
            try:
                if cmd=="ping":
                    if platform.system()=="Windows":
                        out=subprocess.check_output(["ping","-n","4",ip],text=True,timeout=10)
                    else:
                        out=subprocess.check_output(["ping","-c","4",ip],text=True,timeout=10)
                elif cmd=="trace":
                    if platform.system()=="Windows":
                        out=subprocess.check_output(["tracert","-d","-h","15",ip],text=True,timeout=30)
                    else:
                        out=subprocess.check_output(["traceroute","-n","-m","15",ip],text=True,timeout=30)
                elif cmd=="nslookup":
                    out=subprocess.check_output(["nslookup",ip],text=True,timeout=5)
                elif cmd=="arp":
                    if platform.system()=="Windows":
                        out=subprocess.check_output(["arp","-a",ip],text=True,timeout=5)
                    else:
                        out=subprocess.check_output(["arp","-n",ip],text=True,timeout=5)
                elif cmd=="netstat":
                    if platform.system()=="Windows":
                        out=subprocess.check_output(["netstat","-an"],text=True,timeout=5)
                    else:
                        out=subprocess.check_output(["netstat","-an"],text=True,timeout=5)
                elif cmd=="portcheck":
                    results=[]
                    for p in [21,22,23,25,53,80,110,135,139,143,443,445,
                              3306,3389,5432,5900,6379,8080,27017]:
                        st="OPEN ✓" if tcp_open(ip,p,0.8) else "closed"
                        if "OPEN" in st:
                            results.append(f"  Port {p:5d} ({svc_name(p):12s}): {st}")
                    out = "\n".join(results) if results else "No common ports open."
                else:
                    out="Unknown command."
            except subprocess.TimeoutExpired: out="Command timed out."
            except FileNotFoundError as e: out=f"Command not found: {e}"
            except Exception as e: out=str(e)
            self.root.after(0,lambda:self._net_write(out+"\n",append=True))
        threading.Thread(target=do,daemon=True).start()

    def _net_ai_advice(self):
        ip=self.net_cmd_ip.get().strip()
        if not ip: return
        self._net_write(f"\n🤖 Getting AI security advice for {ip}…\n",append=True)
        def do():
            resp=gemini_ask(
                f"Give me the most important network security checks and commands "
                f"to run against device {ip} on a local network. Include:\n"
                "1. Network reconnaissance commands\n2. Port-specific checks\n"
                "3. Security hardening recommendations\nBe concise and practical.",
                "You are a network security expert. Provide practical, actionable advice.")
            self.root.after(0,lambda:self._net_write(resp+"\n",append=True))
        threading.Thread(target=do,daemon=True).start()

    # ══════════════════════════════════════════════════════════════════════
    #  HISTORY PAGE
    # ══════════════════════════════════════════════════════════════════════
    def _page_history(self):
        frame=tk.Frame(self.content,bg=BG)
        tk.Label(frame,text="Scan History",
                 font=("Segoe UI",17,"bold"),fg=ACCENT,bg=BG,pady=14).pack()
        bf=tk.Frame(frame,bg=BG); bf.pack(fill="x",padx=20,pady=(0,10))
        tk.Button(bf,text="🔄 Refresh",bg=CARD_BG,fg=TEXT,
                  font=("Segoe UI",9),bd=0,padx=12,pady=5,cursor="hand2",
                  command=self._load_hist).pack(side="left")
        tk.Button(bf,text="🗑 Clear All",bg=RED,fg="white",
                  font=("Segoe UI",9),bd=0,padx=12,pady=5,cursor="hand2",
                  command=self._clear_hist).pack(side="left",padx=6)
        tk.Button(bf,text="➕ New Scan",bg=ACCENT,fg="#000",
                  font=("Segoe UI",9,"bold"),bd=0,padx=14,pady=5,cursor="hand2",
                  command=self._new_scan).pack(side="left",padx=6)
        tk.Label(bf,text="Double-click to restore full scan session",
                 fg=TEXT_DIM,bg=BG,font=("Segoe UI",9)).pack(side="right")

        cols=("ID","Name","Target","IP","Range","Protocol","Timestamp")
        self.hist_tree=ttk.Treeview(frame,columns=cols,show="headings",
                                     style="History.Treeview")
        for col,w in zip(cols,[40,140,140,120,80,70,150]):
            self.hist_tree.heading(col,text=col)
            self.hist_tree.column(col,width=w)
        vsb2=ttk.Scrollbar(frame,orient="vertical",command=self.hist_tree.yview)
        self.hist_tree.configure(yscrollcommand=vsb2.set)
        self.hist_tree.pack(side="left",fill="both",expand=True,padx=(20,0))
        vsb2.pack(side="left",fill="y",padx=(0,20))
        self.hist_tree.bind("<Double-1>",self._restore)
        self._load_hist()
        return frame

    def _load_hist(self):
        self.hist_tree.delete(*self.hist_tree.get_children())
        for row in db_all_scans():
            self.hist_tree.insert("","end",values=row)

    def _clear_hist(self):
        if messagebox.askyesno("Confirm","Delete all scan history?"):
            db_clear_scans(); self._load_hist()

    def _new_scan(self):
        self.scan_results=[]; self.vuln_results={}
        self.ai_text=""; self.scan_folder=""
        self.log_lines=[]; self.err_lines=[]
        self._open_ct=0; self._err_ct=0
        self.domain_e.delete(0,"end"); self.ip_e.delete(0,"end")
        self.sp_e.delete(0,"end"); self.sp_e.insert(0,"1")
        self.ep_e.delete(0,"end"); self.ep_e.insert(0,"1024")
        self.tree.delete(*self.tree.get_children())
        self.empty_lbl.pack_forget()
        self.status_var.set("Ready — enter target and click ▶ Start Scan")
        self.openct_var.set("Open: 0"); self.errct_var.set("Errors: 0")
        self.curport_var.set("Port: —"); self.prog_var.set(0)
        self.vuln_sum_var.set("Run a scan.")
        self._ai_write("AI report appears here after scanning.")
        self._show("scanner")

    def _restore(self, _e):
        sel=self.hist_tree.selection()
        if not sel: return
        sid=int(self.hist_tree.item(sel[0],"values")[0])
        row=db_one_scan(sid)
        if not row: return
        (_,sname,target,ip,pr,proto,opj,vj,ai,sf,ts,*_)=row
        try: self.scan_results=[tuple(r) for r in json.loads(opj or "[]")]
        except: self.scan_results=[]
        try: self.vuln_results=json.loads(vj or "{}")
        except: self.vuln_results={}
        self.ai_text=ai or ""; self.scan_folder=sf or ""
        self._show("scanner")
        self.domain_e.delete(0,"end"); self.domain_e.insert(0,target or "")
        self.ip_e.delete(0,"end");     self.ip_e.insert(0,ip or "")
        if pr and "-" in str(pr):
            s,e=str(pr).split("-",1)
            self.sp_e.delete(0,"end"); self.sp_e.insert(0,s)
            self.ep_e.delete(0,"end"); self.ep_e.insert(0,e)
        self.tree.delete(*self.tree.get_children())
        self.empty_lbl.pack_forget()
        for r in self.scan_results:
            self.tree.insert("","end",values=r,tags=("open",))
        if self.ai_text: self._ai_write(self.ai_text)
        self._upd_vuln_sum()
        self.status_var.set(f"✅ Restored: '{sname}'  [{ts}]")

    # ══════════════════════════════════════════════════════════════════════
    #  SETTINGS PAGE
    # ══════════════════════════════════════════════════════════════════════
    def _page_settings(self):
        frame=tk.Frame(self.content,bg=BG)
        tk.Label(frame,text="⚙️ Settings",
                 font=("Segoe UI",17,"bold"),fg=ACCENT,bg=BG,pady=14).pack()
        tk.Label(frame,text="Changes are saved immediately and persist across sessions.",
                 font=("Segoe UI",9),fg=TEXT_DIM,bg=BG).pack()

        sf=ScrollFrame(frame,bg=BG); sf.pack(fill="both",expand=True,padx=20,pady=8)
        body=sf.inner

        def section(title):
            tk.Label(body,text=title,fg=ACCENT2,bg=BG,
                     font=("Segoe UI",11,"bold"),anchor="w").pack(
                         fill="x",padx=4,pady=(14,4))
            tk.Frame(body,bg=BORDER,height=1).pack(fill="x",padx=4,pady=(0,8))

        def row_lbl(parent, lbl, hint=""):
            f=tk.Frame(parent,bg=CARD_BG); f.pack(fill="x",pady=4)
            tk.Label(f,text=lbl,fg=TEXT,bg=CARD_BG,
                     font=("Segoe UI",9,"bold"),width=22,anchor="w").pack(side="left",padx=(14,8))
            if hint:
                tk.Label(f,text=hint,fg=TEXT_DIM,bg=CARD_BG,
                         font=("Segoe UI",8)).pack(side="right",padx=14)
            return f

        # ── AI / Gemini
        section("🤖 AI Configuration")
        c1=tk.Frame(body,bg=CARD_BG,padx=14,pady=14); c1.pack(fill="x",padx=4,pady=3)

        tk.Label(c1,text="Google Gemini API Key",fg=TEXT,bg=CARD_BG,
                 font=("Segoe UI",9,"bold")).pack(anchor="w")
        tk.Label(c1,text="Get a free key at: https://makersuite.google.com/app/apikey",
                 fg=ACCENT2,bg=CARD_BG,font=("Segoe UI",8),cursor="hand2").pack(
                     anchor="w",pady=(2,6))
        ak_frame=tk.Frame(c1,bg=CARD_BG); ak_frame.pack(fill="x")
        self.api_key_var=tk.StringVar(value=CFG.get("gemini_api_key",""))
        api_entry=tk.Entry(ak_frame,textvariable=self.api_key_var,
                           bg="#0d1117",fg=TEXT,font=("Segoe UI",10),
                           relief="flat",highlightthickness=1,
                           highlightbackground=BORDER,highlightcolor=ACCENT,
                           show="*",width=48)
        api_entry.pack(side="left",fill="x",expand=True)
        show_var=tk.BooleanVar(value=False)
        def toggle_show():
            api_entry.config(show="" if show_var.get() else "*")
        tk.Checkbutton(ak_frame,text="Show",variable=show_var,
                       bg=CARD_BG,fg=TEXT_DIM,selectcolor=CARD_BG,
                       font=("Segoe UI",9),cursor="hand2",
                       command=toggle_show).pack(side="left",padx=8)

        def test_key():
            key=self.api_key_var.get().strip()
            if not key: messagebox.showwarning("Empty","Enter an API key first."); return
            self.api_status_lbl.config(text="Testing…",fg=YELLOW)
            def do():
                CFG["gemini_api_key"]=key
                r=gemini_ask("Reply with: API_OK","You are a test assistant.")
                ok="API_OK" in r or len(r)>5
                self.root.after(0,lambda:self.api_status_lbl.config(
                    text="✅ Key valid!" if ok else f"❌ {r[:60]}",
                    fg=GREEN if ok else RED))
            threading.Thread(target=do,daemon=True).start()

        tk.Button(c1,text="🔬 Test Key",bg=ACCENT2,fg="#000",
                  font=("Segoe UI",9,"bold"),bd=0,padx=10,pady=4,
                  cursor="hand2",command=test_key).pack(anchor="w",pady=(8,0))
        self.api_status_lbl=tk.Label(c1,text="",fg=TEXT_DIM,bg=CARD_BG,
                                      font=("Segoe UI",9))
        self.api_status_lbl.pack(anchor="w",pady=2)
        tk.Button(c1,text="🌐 Get Free API Key",bg=CARD_BG,fg=ACCENT2,
                  font=("Segoe UI",9),bd=1,relief="solid",padx=10,pady=3,
                  cursor="hand2",
                  command=lambda:webbrowser.open(
                      "https://makersuite.google.com/app/apikey")).pack(
                          anchor="w",pady=(4,0))

        # ── Folders
        section("📁 File & Folder Settings")
        c2=tk.Frame(body,bg=CARD_BG,padx=14,pady=14); c2.pack(fill="x",padx=4,pady=3)
        tk.Label(c2,text="Scans Save Folder:",fg=TEXT,bg=CARD_BG,
                 font=("Segoe UI",9,"bold")).pack(anchor="w")
        sf2=tk.Frame(c2,bg=CARD_BG); sf2.pack(fill="x",pady=(4,8))
        self.folder_var=tk.StringVar(value=CFG.get("scans_folder",SCANS_DIR_DEFAULT))
        fe=tk.Entry(sf2,textvariable=self.folder_var,bg="#0d1117",fg=TEXT,
                    font=("Segoe UI",9),relief="flat",highlightthickness=1,
                    highlightbackground=BORDER,highlightcolor=ACCENT)
        fe.pack(side="left",fill="x",expand=True)
        def browse_folder():
            p=filedialog.askdirectory(initialdir=self.folder_var.get())
            if p: self.folder_var.set(p)
        tk.Button(sf2,text="Browse",bg=CARD_BG,fg=TEXT_DIM,
                  font=("Segoe UI",9),bd=1,relief="solid",padx=10,pady=3,
                  cursor="hand2",command=browse_folder).pack(side="left",padx=6)
        tk.Button(sf2,text="Open",bg=CARD_BG,fg=ACCENT2,
                  font=("Segoe UI",9),bd=1,relief="solid",padx=10,pady=3,
                  cursor="hand2",
                  command=lambda:open_folder(self.folder_var.get())).pack(side="left")

        # auto-save checkboxes
        self.auto_pdf_var=tk.BooleanVar(value=CFG.get("auto_save_pdf",True))
        tk.Checkbutton(c2,text="Auto-save PDF after Automated scan",
                       variable=self.auto_pdf_var,bg=CARD_BG,fg=TEXT,
                       selectcolor=CARD_BG,activebackground=CARD_BG,
                       activeforeground=ACCENT,font=("Segoe UI",9),
                       cursor="hand2").pack(anchor="w",pady=2)
        self.auto_open_var=tk.BooleanVar(value=CFG.get("auto_open_folder",True))
        tk.Checkbutton(c2,text="Auto-open scan folder after Automated scan",
                       variable=self.auto_open_var,bg=CARD_BG,fg=TEXT,
                       selectcolor=CARD_BG,activebackground=CARD_BG,
                       activeforeground=ACCENT,font=("Segoe UI",9),
                       cursor="hand2").pack(anchor="w",pady=2)

        # ── Scan Defaults
        section("🔍 Scan Defaults")
        c3=tk.Frame(body,bg=CARD_BG,padx=14,pady=14); c3.pack(fill="x",padx=4,pady=3)
        row=tk.Frame(c3,bg=CARD_BG); row.pack(fill="x",pady=4)
        tk.Label(row,text="Default Protocol:",fg=TEXT,bg=CARD_BG,
                 font=("Segoe UI",9),width=20,anchor="w").pack(side="left")
        self.def_proto_var=tk.StringVar(value=CFG.get("default_proto","TCP"))
        ttk.Combobox(row,textvariable=self.def_proto_var,
                     values=["TCP","UDP","BOTH"],width=10,
                     state="readonly").pack(side="left",padx=8)
        row2=tk.Frame(c3,bg=CARD_BG); row2.pack(fill="x",pady=4)
        tk.Label(row2,text="Default Scan Mode:",fg=TEXT,bg=CARD_BG,
                 font=("Segoe UI",9),width=20,anchor="w").pack(side="left")
        self.def_mode_var=tk.StringVar(value=CFG.get("default_mode","Traditional"))
        ttk.Combobox(row2,textvariable=self.def_mode_var,
                     values=["Traditional","Automated"],width=12,
                     state="readonly").pack(side="left",padx=8)
        row3=tk.Frame(c3,bg=CARD_BG); row3.pack(fill="x",pady=4)
        tk.Label(row3,text="Default Assessor Name:",fg=TEXT,bg=CARD_BG,
                 font=("Segoe UI",9),width=20,anchor="w").pack(side="left")
        self.def_assessor_var=tk.StringVar(value=CFG.get("assessor_name","Security Analyst"))
        tk.Entry(row3,textvariable=self.def_assessor_var,bg="#0d1117",fg=TEXT,
                 font=("Segoe UI",9),relief="flat",highlightthickness=1,
                 highlightbackground=BORDER,highlightcolor=ACCENT,
                 width=28).pack(side="left",padx=8)

        # ── Save button
        def _save_all():
            CFG["gemini_api_key"]  = self.api_key_var.get().strip()
            CFG["scans_folder"]    = self.folder_var.get().strip()
            CFG["auto_save_pdf"]   = self.auto_pdf_var.get()
            CFG["auto_open_folder"]= self.auto_open_var.get()
            CFG["default_proto"]   = self.def_proto_var.get()
            CFG["default_mode"]    = self.def_mode_var.get()
            CFG["assessor_name"]   = self.def_assessor_var.get()
            save_cfg(CFG)
            # update live vars
            self.assessor_var.set(CFG["assessor_name"])
            self.proto_var.set(CFG["default_proto"])
            self.mode_var.set(CFG["default_mode"])
            os.makedirs(CFG["scans_folder"],exist_ok=True)
            messagebox.showinfo("Saved","Settings saved successfully! ✅")

        tk.Button(body,text="💾  Save All Settings",
                  bg=ACCENT,fg="#000",font=("Segoe UI",11,"bold"),
                  bd=0,padx=24,pady=10,cursor="hand2",
                  command=_save_all).pack(pady=16)
        return frame

    # ══════════════════════════════════════════════════════════════════════
    #  TOOLS PAGE
    # ══════════════════════════════════════════════════════════════════════
    def _page_tools(self):
        frame=tk.Frame(self.content,bg=BG)
        tk.Label(frame,text="🛠 OSINT & Security Tools",
                 font=("Segoe UI",17,"bold"),fg=ACCENT,bg=BG,pady=12).pack()

        sf=ScrollFrame(frame,bg=BG); sf.pack(fill="both",expand=True,padx=20,pady=4)
        body=sf.inner

        TOOLS=[
            ("🔍 DNS & Domain",""),
            ("NSLookup.io",       "https://www.nslookup.io",          "DNS: A, MX, NS, TXT, CNAME"),
            ("WHOIS DomainTools", "https://whois.domaintools.com",     "Domain ownership & history"),
            ("ViewDNS.info",      "https://viewdns.info",              "Reverse DNS, IP history, ping"),
            ("HackerTarget",      "https://hackertarget.com",          "DNS lookup, subnet tools"),
            ("DNSDumpster",       "https://dnsdumpster.com",           "Domain recon map"),
            ("Shrewdeye",         "https://shrewdeye.app",             "Passive subdomain enumeration"),

            ("🌐 IP & Geo",""),
            ("IPGeolocation.io",  "https://ipgeolocation.io",          "IP: country, ISP, ASN, coords"),
            ("Reverse IP",        "https://viewdns.info/reverseip",    "Domains hosted on same IP"),
            ("Shodan.io",         "https://www.shodan.io",             "Internet device search engine"),
            ("Censys",            "https://search.censys.io",          "Internet-wide host data"),

            ("🕷 Web Tech",""),
            ("Wappalyzer",        "https://www.wappalyzer.com",        "CMS, frameworks, analytics"),
            ("BuiltWith",         "https://builtwith.com",             "Full tech stack profiler"),
            ("Wayback Machine",   "https://web.archive.org",           "Historical website snapshots"),
            ("Security Headers",  "https://securityheaders.com",       "HTTP security header analysis"),

            ("💀 Exploits & CVE",""),
            ("Exploit-DB",        "https://www.exploit-db.com",        "Public exploit database"),
            ("NVD CVE Search",    "https://nvd.nist.gov/vuln/search",  "NIST vulnerability database"),
            ("CVE Mitre",         "https://cve.mitre.org",             "CVE identifiers"),
            ("Google Dorks",      "https://www.exploit-db.com/google-hacking-database","Google hacking operators"),

            ("📧 OSINT",""),
            ("Hunter.io",         "https://hunter.io",                 "Find email addresses"),
            ("Maltego CE",        "https://www.maltego.com/maltego-community","OSINT graph analysis"),
            ("SSL Labs",          "https://www.ssllabs.com/ssltest",   "SSL/TLS deep analysis"),
            ("Observatory",       "https://observatory.mozilla.org",   "Website security scan"),
        ]

        row_frame=None; col_count=0
        for item in TOOLS:
            if len(item)==2 and item[1]=="":
                tk.Label(body,text=item[0],fg=ACCENT2,bg=BG,
                         font=("Segoe UI",11,"bold"),anchor="w").pack(
                             fill="x",padx=4,pady=(12,4))
                tk.Frame(body,bg=BORDER,height=1).pack(fill="x",padx=4,pady=(0,6))
                row_frame=tk.Frame(body,bg=BG); row_frame.pack(fill="x"); col_count=0; continue
            name,url,desc=item
            c=tk.Frame(row_frame,bg=CARD_BG,padx=12,pady=10)
            c.grid(row=0,column=col_count,padx=5,pady=4,sticky="nsew")
            row_frame.columnconfigure(col_count,weight=1); col_count+=1
            tk.Label(c,text=name,fg=ACCENT,bg=CARD_BG,
                     font=("Segoe UI",10,"bold"),anchor="w").pack(anchor="w")
            tk.Label(c,text=desc,fg=TEXT_DIM,bg=CARD_BG,
                     font=("Segoe UI",8),wraplength=190,justify="left").pack(anchor="w",pady=(2,6))
            tk.Button(c,text="Open →",bg=ACCENT2,fg="#000",
                      font=("Segoe UI",9,"bold"),bd=0,padx=8,pady=3,cursor="hand2",
                      command=lambda u=url:webbrowser.open(u)).pack(anchor="w")
            if col_count>=3:
                row_frame=tk.Frame(body,bg=BG); row_frame.pack(fill="x"); col_count=0

        # quick lookup widget
        lk=tk.Frame(frame,bg=CARD_BG,padx=20,pady=12)
        lk.pack(padx=20,fill="x",pady=(6,14))
        tk.Label(lk,text="⚡ Quick Domain Lookup",fg=ACCENT,bg=CARD_BG,
                 font=("Segoe UI",11,"bold")).pack(anchor="w")
        lr=tk.Frame(lk,bg=CARD_BG); lr.pack(fill="x",pady=8)
        self.td_e=tk.Entry(lr,bg="#0d1117",fg=TEXT,font=("Segoe UI",10),
                            relief="flat",highlightthickness=1,
                            highlightbackground=BORDER,highlightcolor=ACCENT,width=28)
        self.td_e.insert(0,"example.com"); self.td_e.pack(side="left")
        self.td_lbl=tk.Label(lk,text="",fg=GREEN,bg=CARD_BG,font=("Segoe UI",10,"bold"))
        self.td_lbl.pack(anchor="w")
        for txt,url_t in [("NSLookup","https://www.nslookup.io/dns-records/{d}/"),
                           ("WHOIS","https://whois.domaintools.com/{d}"),
                           ("Subdomains","https://shrewdeye.app/?q={d}"),
                           ("Shodan","https://www.shodan.io/search?query={d}")]:
            tk.Button(lr,text=txt,bg=CARD_BG,fg=ACCENT2,font=("Segoe UI",9),
                      bd=1,relief="solid",padx=7,pady=3,cursor="hand2",
                      command=lambda u=url_t:webbrowser.open(
                          u.replace("{d}",self.td_e.get().strip()))
                      ).pack(side="left",padx=3)
        def do_res():
            d=self.td_e.get().strip(); ip=resolve(d)
            self.td_lbl.config(text=f"{d} → {ip}" if ip else "Cannot resolve",
                                fg=GREEN if ip else RED)
        tk.Button(lr,text="Resolve IP",bg=ACCENT,fg="#000",
                  font=("Segoe UI",9,"bold"),bd=0,padx=10,pady=3,cursor="hand2",
                  command=do_res).pack(side="left",padx=8)
        return frame

    # ══════════════════════════════════════════════════════════════════════
    #  ABOUT PAGE
    # ══════════════════════════════════════════════════════════════════════
    def _page_about(self):
        frame=tk.Frame(self.content,bg=BG)
        hf=tk.Frame(frame,bg="#0f1824",pady=16); hf.pack(fill="x")
        tk.Label(hf,text="🏢 SUPRAJA TECHNOLOGIES",
                 font=("Segoe UI",20,"bold"),fg=ACCENT,bg="#0f1824").pack()
        tk.Label(hf,text="a unit of CHSMRLSS Technologies Pvt. Ltd.",
                 font=("Segoe UI",10),fg=TEXT_DIM,bg="#0f1824").pack(pady=2)
        tk.Label(hf,text="Vijayawada, Andhra Pradesh, India  |  4.8 ⭐ Google",
                 font=("Segoe UI",10),fg=YELLOW,bg="#0f1824").pack()
        tk.Button(hf,text="🌐 www.suprajatechnologies.com",
                  bg="#0f1824",fg=ACCENT2,font=("Segoe UI",10,"bold"),bd=0,cursor="hand2",
                  command=lambda:webbrowser.open("https://www.suprajatechnologies.com")).pack(pady=4)

        sf=ScrollFrame(frame,bg=BG); sf.pack(fill="both",expand=True,padx=20,pady=8)
        body=sf.inner

        def section(title):
            tk.Label(body,text=title,fg=ACCENT2,bg=BG,font=("Segoe UI",12,"bold"),
                     anchor="w").pack(fill="x",padx=4,pady=(12,4))
            tk.Frame(body,bg=BORDER,height=1).pack(fill="x",padx=4,pady=(0,6))

        def card(lines,fg_=TEXT):
            c=tk.Frame(body,bg=CARD_BG,padx=16,pady=10); c.pack(fill="x",padx=4,pady=3)
            for l in lines:
                tk.Label(c,text=l,fg=fg_,bg=CARD_BG,font=("Segoe UI",9),
                         anchor="w",wraplength=880,justify="left").pack(anchor="w",pady=1)

        section("📌 About")
        card(["Supraja Technologies is a leading Knowledge and Technical Solutions Provider.",
              "Foundation pillars: Innovation, Information and Intelligence.",
              "Operates as: Technology Service Provider (Corporate Consulting) + Training Organization (Ed-Tech).",
              "🔬 24×7 R&D — Supraja Technologies Cyber Security Cell",
              "📍 Vijayawada, Andhra Pradesh, India   |   👨‍💼 CEO: Mr. Santosh Chaluvadi"])

        section("🏆 Achievements")
        card(["📖 LIMCA BOOK OF RECORDS 2017 — 50-hour Nonstop Marathon Workshop on Ethical Hacking",
              "🏅 Top 50 Tech Companies 2019 — InterCon Dubai, UAE",
              "🛡 CoE @ Ramco Institute of Technology, Rajapalayam (17 Aug 2024)",
              "🛡 CoE @ SRM University, Ramapuram, Chennai (18 Sep 2024)",
              "🛡 CoE @ St. Joseph's Institute of Technology, Chennai (20 Nov 2024)",
              "🎬 Anti-Piracy Solution: kills up to 35% online piracy for Tollywood film industry"],GREEN)

        section("📚 Training Programs")
        pf=tk.Frame(body,bg=BG); pf.pack(fill="x",padx=4,pady=3)
        for i,(title,items) in enumerate([
            ("🎓 Classroom",["Summer (30–45 days)","Winter (10–15 days)","Weekend (2 days)","1/3/6 Month"]),
            ("🏫 On-site",  ["Value Added Courses","Faculty Dev Programs","Govt Agencies","Corporate"]),
            ("💼 Internship",["Students 30/45/60 days","Graduates 6 months","Cyber Focus"]),
            ("🔬 Workshop", ["Engineering Colleges","Corporates / MNCs","Govt Organizations","Hackathons"]),
        ]):
            c=tk.Frame(pf,bg=CARD_BG,padx=12,pady=10)
            c.grid(row=0,column=i,padx=4,pady=3,sticky="nsew")
            pf.columnconfigure(i,weight=1)
            tk.Label(c,text=title,fg=ACCENT,bg=CARD_BG,font=("Segoe UI",10,"bold")).pack(anchor="w")
            for itm in items:
                tk.Label(c,text=f"• {itm}",fg=TEXT,bg=CARD_BG,
                         font=("Segoe UI",8),anchor="w").pack(anchor="w")

        section("🌟 Why Supraja")
        card(["✔ 68,500+ students trained","✔ Proven quality cybersecurity services",
              "✔ Training partners of recognized institutions",
              "✔ Hands-on sessions with Study Material + Toolkit","✔ Self-prepared Cyber Security Cell"])

        section("🔗 Connect")
        lf=tk.Frame(body,bg=CARD_BG,padx=16,pady=10); lf.pack(fill="x",padx=4,pady=3)
        for lbl,url in [("🌐 Website","https://www.suprajatechnologies.com"),
                        ("⭐ Google Reviews","https://bit.ly/SuprajaGoogle"),
                        ("📸 CEO Instagram","https://www.instagram.com/chaluvadisantosh/")]:
            tk.Button(lf,text=lbl,bg=CARD_BG,fg=ACCENT2,font=("Segoe UI",10),
                      bd=0,cursor="hand2",command=lambda u=url:webbrowser.open(u)).pack(anchor="w",pady=3)

        section("🔧 System")
        gk="✓ Key loaded" if get_api_key() else "✗ Not set — go to ⚙️ Settings"
        card([f"Hostname : {socket.gethostname()}",
              f"Local IP : {local_ip()}",
              f"Platform : {platform.system()} {platform.release()}",
              f"Python   : {sys.version.split()[0]}",
              f"Gemini   : {gk}",
              f"ReportLab: {'Installed ✓' if REPORTLAB_OK else 'pip install reportlab'}",
              f"Scans Dir: {scans_dir()}"])

        # domain lookup
        lc=tk.Frame(frame,bg=CARD_BG,padx=20,pady=10)
        lc.pack(padx=20,fill="x",pady=(0,12))
        tk.Label(lc,text="Quick Lookup:",fg=ACCENT,bg=CARD_BG,
                 font=("Segoe UI",10,"bold")).pack(side="left",padx=(0,10))
        self.abt_e=tk.Entry(lc,bg="#0d1117",fg=TEXT,font=("Segoe UI",10),
                             relief="flat",highlightthickness=1,
                             highlightbackground=BORDER,highlightcolor=ACCENT,width=28)
        self.abt_e.insert(0,"google.com"); self.abt_e.pack(side="left")
        self.abt_lbl=tk.Label(lc,text="",fg=GREEN,bg=CARD_BG,font=("Segoe UI",10,"bold"))
        self.abt_lbl.pack(side="left",padx=12)
        tk.Button(lc,text="Lookup",bg=ACCENT2,fg="#000",
                  font=("Segoe UI",9,"bold"),bd=0,padx=10,pady=4,cursor="hand2",
                  command=lambda:self.abt_lbl.config(
                      text=f"{self.abt_e.get().strip()} → "
                           f"{resolve(self.abt_e.get().strip()) or 'N/A'}")).pack(side="left")
        return frame

    # ══════════════════════════════════════════════════════════════════════
    #  PROMPT
    # ══════════════════════════════════════════════════════════════════════
    def _parse_prompt(self):
        raw=self.prompt_e.get().strip()
        from __main__ import parse_prompt as _pp
        target,sp,ep=_pp(raw)
        if target: self._fill(target,sp,ep)

    def _ai_parse(self):
        raw=self.prompt_e.get().strip()
        if not raw or "e.g." in raw: return
        self.status_var.set("🤖 AI parsing prompt…")
        def do():
            resp=gemini_ask(
                f"Extract scan parameters from: {raw}\n"
                "Reply ONLY as JSON: {\"target\":\"...\",\"start\":N,\"end\":N}",
                "You are a network tool. Extract only scan params.")
            try:
                m=re.search(r'\{[^}]+\}',resp)
                if m:
                    d=json.loads(m.group())
                    self.root.after(0,lambda:self._fill(
                        d.get("target"),d.get("start"),d.get("end"))); return
            except: pass
            self.root.after(0,self._parse_prompt)
        threading.Thread(target=do,daemon=True).start()

    def _fill(self,target,sp,ep):
        if not target: return
        self.domain_e.delete(0,"end"); self.ip_e.delete(0,"end")
        if re.match(r"\d+\.\d+\.\d+\.\d+",str(target)):
            self.ip_e.insert(0,str(target))
        else:
            self.domain_e.insert(0,str(target))
            ir=resolve(str(target))
            if ir: self.ip_e.insert(0,ir)
        if sp: self.sp_e.delete(0,"end"); self.sp_e.insert(0,str(sp))
        if ep: self.ep_e.delete(0,"end"); self.ep_e.insert(0,str(ep))
        self.status_var.set(f"Filled → {target}  ports {sp}–{ep}")

    # ══════════════════════════════════════════════════════════════════════
    #  SCAN FLOW
    # ══════════════════════════════════════════════════════════════════════
    def _resolve(self):
        d=self.domain_e.get().strip()
        if not d: return
        ip=resolve(d)
        if ip:
            self.ip_e.delete(0,"end"); self.ip_e.insert(0,ip)
            self.status_var.set(f"Resolved {d} → {ip}")
        else:
            messagebox.showerror("DNS Error",f"Cannot resolve: {d}")

    def _start(self):
        if self.scanning: return
        domain=self.domain_e.get().strip(); ip=self.ip_e.get().strip()
        if domain and not ip:
            ip=resolve(domain)
            if ip: self.ip_e.delete(0,"end"); self.ip_e.insert(0,ip)
        if not ip:
            messagebox.showerror("Error","Enter IP or Domain."); return
        try:
            sp=int(self.sp_e.get()); ep=int(self.ep_e.get())
            if sp<0 or ep>65535 or sp>ep: raise ValueError
        except ValueError:
            messagebox.showerror("Error","Ports 0–65535, Start ≤ End."); return
        if self.allports_var.get(): sp,ep=0,65535

        self.scanning=True; self.scan_results=[]; self.vuln_results={}
        self.nmap_cmds={}; self.log_lines=[]; self.err_lines=[]
        self._open_ct=0; self._err_ct=0
        self.scan_folder=make_folder(domain or ip)
        self.tree.delete(*self.tree.get_children())
        self.empty_lbl.pack_forget()
        self.scan_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.prog_var.set(0); self.openct_var.set("Open: 0")
        self.errct_var.set("Errors: 0"); self.vuln_sum_var.set("Scanning…")
        self.status_var.set(f"Starting scan on {ip}…")

        threading.Thread(target=self._run,
                         args=(ip,sp,ep,self.proto_var.get(),
                               domain,self.mode_var.get()),daemon=True).start()

    def _stop(self):
        self.scanning=False; self.status_var.set("Scan stopped.")
        self.scan_btn.config(state="normal"); self.stop_btn.config(state="disabled")

    def _run(self,ip,sp,ep,proto,domain,mode):
        protocols=(["TCP","UDP"] if proto=="BOTH" else [proto.upper()])
        total=(ep-sp+1)*len(protocols); done=0; open_pairs=[]

        self.root.after(0,lambda:self.status_var.set(
            f"Phase 1/2 — Sweeping {ip} {sp}–{ep}…"))

        def chk(port,pproto):
            return (port,pproto) if (tcp_open(ip,port) if pproto=="TCP"
                                      else udp_probe(ip,port)) else None

        with concurrent.futures.ThreadPoolExecutor(max_workers=min(512,max(total,1))) as ex:
            futs={ex.submit(chk,p,pp):(p,pp)
                  for p in range(sp,ep+1) for pp in protocols}
            for fut in concurrent.futures.as_completed(futs):
                if not self.scanning: break
                done+=1; p,pp=futs[fut]
                self.root.after(0,lambda v=60*done/total:self.prog_var.set(v))
                self.root.after(0,lambda po=p,pr=pp:self.curport_var.set(f"Port {po}({pr})"))
                try:
                    r=fut.result()
                    if r: open_pairs.append(r)
                except Exception as e:
                    self.err_lines.append(f"Port {p}: {e}"); self._err_ct+=1

        if not open_pairs:
            self.root.after(0,lambda:self._finish(ip,domain,[],sp,ep,proto,mode))
            return

        open_pairs.sort()
        self.root.after(0,lambda:self.status_var.set(
            f"Phase 2/2 — Banners from {len(open_pairs)} open ports…"))

        rows=[]
        for i,(port,pproto) in enumerate(open_pairs):
            if not self.scanning: break
            self.root.after(0,lambda v=60+40*i/len(open_pairs):self.prog_var.set(v))
            try:
                banner=grab_banner(ip,port); name=svc_name(port,pproto.lower())
                row=(port,name,pproto,"open",banner or "—"); rows.append(row)
                self._open_ct+=1; oc=self._open_ct
                self.root.after(0,lambda r=row,c=oc:(
                    self.tree.insert("","end",values=r,tags=("open",)),
                    self.openct_var.set(f"Open: {c}")))
            except Exception as e:
                self.err_lines.append(f"Banner {port}: {e}")
                self._err_ct+=1
                self.root.after(0,lambda c=self._err_ct:self.errct_var.set(f"Errors: {c}"))

        self.root.after(0,lambda:self.prog_var.set(100))
        self.root.after(0,lambda:self._finish(ip,domain,rows,sp,ep,proto,mode))

    def _finish(self,ip,domain,rows,sp,ep,proto,mode):
        self.scan_results=rows; self.scanning=False
        self.scan_btn.config(state="normal"); self.stop_btn.config(state="disabled")

        if not rows:
            self.empty_lbl.config(text=f"No open ports on {ip} ({sp}–{ep})")
            self.empty_lbl.pack(pady=8)
            self.status_var.set("Scan complete — 0 open ports.")
            self._save_artifacts(domain or ip,rows,{},""); return

        # vulns — OPEN PORTS ONLY
        self.vuln_results={}; crit=hi=med=lo=0
        for r in rows:
            port,name,pproto,_,banner=r
            vulns=self._assess(port,name,pproto,banner)
            self.vuln_results[str(port)]={
                "service":name,"protocol":pproto,
                "banner":banner,"vulnerabilities":vulns}
            cmds=NMAP_CMDS.get(port,[f"nmap -sV -p {port} {{ip}}"])
            self.nmap_cmds[str(port)]=[c.replace("{ip}",ip) for c in cmds]
            for v in vulns:
                s=v["severity"]
                if s=="Critical": crit+=1
                elif s=="High":   hi+=1
                elif s=="Medium": med+=1
                else:             lo+=1

        summary=f"Critical:{crit}  High:{hi}  Medium:{med}  Low:{lo}"
        self.vuln_sum_var.set(summary)
        self.status_var.set(f"✅ Done — {len(rows)} open port(s) | {summary}")
        self._upd_vuln_sum()

        if mode=="Automated":
            self._ai_write("🤖 Automated mode running: Nmap AI + Full Report + Auto-save…")
            threading.Thread(target=self._auto_pipeline,args=(ip,domain),daemon=True).start()
        else:
            self._save_artifacts(domain or ip,rows,self.vuln_results,"")

    def _auto_pipeline(self,ip,domain):
        """Full automated pipeline with auto PDF save and folder open."""
        # Step 1: Nmap commands via AI
        self.root.after(0,lambda:self.status_var.set("🤖 AI generating Nmap commands…"))
        nmap_advice=gemini_ask(
            f"For penetration test on {ip}, open ports:\n"+
            "\n".join(f"  Port {p}({d['service']}): {', '.join(self.nmap_cmds.get(p,[]))}"
                      for p,d in self.vuln_results.items())+
            "\n\nProvide complete Nmap commands with scripts for each port. "
            "Include version scan, vulnerability scan, brute-force check.",
            "You are an expert penetration tester.")

        write_file(self.scan_folder,"nmap_commands.txt",
                   f"AI Nmap Commands for {ip}\n\n{nmap_advice}")

        # Step 2: Full AI report
        self.root.after(0,lambda:self.status_var.set("🤖 Generating full AI report…"))
        report=self._build_report(ip,nmap_advice)
        self.ai_text=report
        self.root.after(0,lambda:self._ai_write(report))
        self._save_artifacts(domain or ip,self.scan_results,self.vuln_results,report)

        # Step 3: Auto-save PDF
        if CFG.get("auto_save_pdf",True):
            self.root.after(0,lambda:self.status_var.set("💾 Auto-saving PDF report…"))
            pdf_path=os.path.join(self.scan_folder,"security_report.pdf")
            mit_path=os.path.join(self.scan_folder,"mitigation_report.pdf")
            self.root.after(100,lambda:self._build_pdf(pdf_path,"summary",silent=True))
            self.root.after(200,lambda:self._build_pdf(mit_path,"mitigation",silent=True))

        # Step 4: Open folder
        if CFG.get("auto_open_folder",True):
            self.root.after(500,lambda:open_folder(self.scan_folder))

        self.root.after(600,lambda:self.status_var.set(
            f"✅ Automated pipeline complete! Folder: {self.scan_folder}"))

    def _assess(self,port,service,proto,banner)->list:
        vulns=list(VULN_DB.get(port,[]))
        if not vulns:
            vulns.append({"severity":"Info","cve":"N/A","cvss":"0.0",
                          "description":f"Port {port} ({service}) is open",
                          "risk":"Service is accessible from external network",
                          "mitigation":"Verify this port should be publicly accessible"})
        if banner:
            bl=banner.lower()
            for pat,desc,sev,cve,cvss in [
                ("apache/2.2","Outdated Apache 2.2","High","CVE-2017-7679","7.5"),
                ("nginx/1.0","Outdated Nginx 1.0","High","Multiple","7.5"),
                ("nginx/1.1","Outdated Nginx 1.1","High","Multiple","7.5"),
                ("openssh/6.","Outdated OpenSSH 6.x","High","Multiple","7.8"),
                ("php/5.","EOL PHP 5.x","Critical","Multiple","9.8"),
                ("php/7.0","EOL PHP 7.0","High","Multiple","7.5"),
                ("tomcat/7.","Outdated Tomcat 7","High","Multiple","7.5"),
            ]:
                if pat in bl:
                    vulns.append({"severity":sev,"cve":cve,"cvss":cvss,
                                  "description":f"Banner reveals {desc}: {banner[:60]}",
                                  "risk":"Known unpatched vulnerabilities",
                                  "mitigation":"Upgrade immediately"})
        return vulns

    def _upd_vuln_sum(self):
        if not self.vuln_results: return
        crit=hi=med=lo=0
        for d in self.vuln_results.values():
            for v in d.get("vulnerabilities",[]):
                s=v["severity"]
                if s=="Critical": crit+=1
                elif s=="High":   hi+=1
                elif s=="Medium": med+=1
                else:             lo+=1
        self.vuln_sum_var.set(f"Critical:{crit}  High:{hi}  Medium:{med}  Low:{lo}")

    def _save_artifacts(self,target,rows,vulns,ai):
        sf=self.scan_folder
        if not sf: return
        write_file(sf,"logs.txt","\n".join(self.log_lines))
        write_file(sf,"errors.txt","\n".join(self.err_lines))
        write_file(sf,"scan_info.json",json.dumps(
            {"target":target,"timestamp":datetime.datetime.now().isoformat(),
             "open_ports":len(rows),"assessor":self.assessor_var.get()},indent=2))
        write_file(sf,"open_ports.json",
                   json.dumps([[r[0],r[1],r[2],r[3],r[4]] for r in rows],indent=2))
        write_file(sf,"vulnerabilities.json",json.dumps(vulns,indent=2))
        write_file(sf,"nmap_commands.json",json.dumps(self.nmap_cmds,indent=2))
        if ai: write_file(sf,"ai_report.txt",ai)
        name=f"{target}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        try:
            db_save_scan(name,target,self.ip_e.get(),
                         f"{self.sp_e.get()}-{self.ep_e.get()}",
                         self.proto_var.get(),rows,vulns,ai,sf,
                         self.assessor_var.get())
            self.root.after(0,self._load_hist)
        except Exception as e: self.err_lines.append(f"DB: {e}")

    # ── Vuln panel
    def _show_vulns(self):
        if not self.vuln_results:
            messagebox.showinfo("No Data","Run a scan first."); return
        win=tk.Toplevel(self.root)
        win.title("Vulnerability Assessment"); win.geometry("860x580"); win.configure(bg=BG)
        tk.Label(win,text="🔴 Full Vulnerability Assessment",
                 font=("Segoe UI",14,"bold"),fg=RED,bg=BG,pady=10).pack()
        cols=("Port","Service","Severity","CVSS","CVE","Description")
        tv=ttk.Treeview(win,columns=cols,show="headings",style="Vuln.Treeview")
        for col,w in zip(cols,[60,90,80,55,130,0]):
            tv.heading(col,text=col); tv.column(col,width=w,stretch=(col=="Description"))
        for sev,cfg in SEV_CLR.items(): tv.tag_configure(sev,foreground=cfg)
        vsb3=ttk.Scrollbar(win,orient="vertical",command=tv.yview)
        tv.configure(yscrollcommand=vsb3.set)
        tv.pack(side="left",fill="both",expand=True,padx=(20,0),pady=(0,20))
        vsb3.pack(side="left",fill="y",padx=(0,20),pady=(0,20))
        for ps,d in self.vuln_results.items():
            for v in d["vulnerabilities"]:
                s=v["severity"]
                tv.insert("","end",values=(ps,d["service"],s,
                           v.get("cvss","N/A"),v["cve"],v["description"]),tags=(s,))

    # ── Port detail
    def _on_dbl(self,_e):
        sel=self.tree.selection()
        if sel: self._detail(self.tree.item(sel[0],"values"))

    def _detail(self,data):
        port=int(data[0]); service=data[1]
        ip=self.ip_e.get() or self.domain_e.get()
        win=tk.Toplevel(self.root)
        win.title(f"Port {port} / {service.upper()} — {ip}")
        win.geometry("760x680"); win.configure(bg=BG)
        tk.Label(win,text=f"Port {port}  /  {service.upper()}",
                 font=("Segoe UI",14,"bold"),fg=ACCENT,bg=BG,pady=10).pack()
        tk.Label(win,text=f"Target: {ip}  |  Proto: {data[2]}  |  Status: {data[3]}",
                 font=("Segoe UI",9),fg=TEXT_DIM,bg=BG).pack()
        tk.Label(win,text=f"Banner: {data[4] or 'N/A'}",
                 font=("Segoe UI",9),fg=YELLOW,bg=BG,pady=2).pack()
        tk.Frame(win,bg=BORDER,height=1).pack(fill="x",padx=20,pady=8)

        nb=ttk.Notebook(win); nb.pack(fill="both",expand=True,padx=20,pady=(0,6))

        # Tab 1: Checks
        t1=tk.Frame(nb,bg=BG); nb.add(t1,text="  Security Checks  ")
        cv=tk.Canvas(t1,bg=BG,bd=0,highlightthickness=0)
        scb_t=ttk.Scrollbar(t1,orient="vertical",command=cv.yview)
        inn=tk.Frame(cv,bg=BG)
        inn.bind("<Configure>",lambda e:cv.configure(scrollregion=cv.bbox("all")))
        cv.create_window((0,0),window=inn,anchor="nw")
        cv.configure(yscrollcommand=scb_t.set)
        cv.pack(side="left",fill="both",expand=True); scb_t.pack(side="right",fill="y")
        cv.bind("<MouseWheel>",lambda e:cv.yview_scroll(int(-1*(e.delta/120)),"units"))

        out_box=scrolledtext.ScrolledText(win,height=7,bg=SIDEBAR_BG,fg=GREEN,
                                           font=("Consolas",8),bd=0,padx=8,pady=6)
        out_box.pack(fill="x",padx=20,pady=(0,6)); win._out=out_box

        for clabel,cmd_t in MANUAL_CHECKS.get(port,DEFAULT_CHECKS):
            cmd=cmd_t.replace("{ip}",ip).replace("{port}",str(port)).replace("{service}",service)
            row=tk.Frame(inn,bg=CARD_BG,pady=6,padx=12); row.pack(fill="x",pady=2)
            tk.Label(row,text=f"  {clabel}",fg=TEXT,bg=CARD_BG,
                     font=("Segoe UI",9,"bold"),width=24,anchor="w").pack(side="left")
            tk.Label(row,text=cmd,fg=ACCENT2,bg=CARD_BG,
                     font=("Consolas",8),anchor="w").pack(side="left",padx=8)
            tk.Button(row,text="Copy",bg=CARD_BG,fg=TEXT_DIM,
                      font=("Segoe UI",8),bd=1,relief="solid",padx=6,pady=2,
                      cursor="hand2",command=lambda c=cmd:self._copy(c)).pack(side="right",padx=(4,0))
            tk.Button(row,text="Test",bg=ACCENT,fg="#000",
                      font=("Segoe UI",8,"bold"),bd=0,padx=8,pady=2,cursor="hand2",
                      command=lambda p=port,l=clabel,w=win:self._run_check(ip,p,l,w)).pack(side="right")

        # Tab 2: Vulnerabilities
        t2=tk.Frame(nb,bg=BG); nb.add(t2,text="  Vulnerabilities  ")
        vlst=self.vuln_results.get(str(port),{}).get("vulnerabilities",[])
        if not vlst:
            tk.Label(t2,text="Run a scan first.",fg=TEXT_DIM,bg=BG,
                     font=("Segoe UI",11)).pack(pady=30)
        else:
            vc=tk.Canvas(t2,bg=BG,bd=0,highlightthickness=0)
            vs=ttk.Scrollbar(t2,orient="vertical",command=vc.yview)
            vi=tk.Frame(vc,bg=BG)
            vi.bind("<Configure>",lambda e:vc.configure(scrollregion=vc.bbox("all")))
            vc.create_window((0,0),window=vi,anchor="nw")
            vc.configure(yscrollcommand=vs.set)
            vc.pack(side="left",fill="both",expand=True); vs.pack(side="right",fill="y")
            vc.bind("<MouseWheel>",lambda e:vc.yview_scroll(int(-1*(e.delta/120)),"units"))
            for vuln in vlst:
                sev=vuln["severity"]; c=SEV_CLR.get(sev,TEXT)
                vrow=tk.Frame(vi,bg=CARD_BG,pady=8,padx=14)
                vrow.pack(fill="x",pady=3,padx=6)
                tk.Label(vrow,text=f"[{sev}]  CVSS:{vuln.get('cvss','N/A')}",
                         fg=c,bg=CARD_BG,font=("Segoe UI",9,"bold")).grid(row=0,column=0,sticky="w")
                tk.Label(vrow,text=vuln["cve"],fg=PURPLE,bg=CARD_BG,
                         font=("Segoe UI",9)).grid(row=0,column=1,sticky="w",padx=10)
                tk.Label(vrow,text=vuln["description"],fg=TEXT,bg=CARD_BG,
                         font=("Segoe UI",9),wraplength=580,justify="left").grid(
                             row=1,column=0,columnspan=2,sticky="w",pady=2)
                tk.Label(vrow,text=f"⚠ Risk: {vuln['risk']}",fg=YELLOW,bg=CARD_BG,
                         font=("Segoe UI",8),wraplength=580,justify="left").grid(
                             row=2,column=0,columnspan=2,sticky="w")
                tk.Label(vrow,text=f"✅ Fix: {vuln['mitigation']}",fg=GREEN,bg=CARD_BG,
                         font=("Segoe UI",8,"bold"),wraplength=580,justify="left").grid(
                             row=3,column=0,columnspan=2,sticky="w")

        # Tab 3: Nmap AI
        t3=tk.Frame(nb,bg=BG); nb.add(t3,text="  🤖 Nmap AI  ")
        ncv=tk.Canvas(t3,bg=BG,bd=0,highlightthickness=0)
        ncs=ttk.Scrollbar(t3,orient="vertical",command=ncv.yview)
        nci=tk.Frame(ncv,bg=BG)
        nci.bind("<Configure>",lambda e:ncv.configure(scrollregion=ncv.bbox("all")))
        ncv.create_window((0,0),window=nci,anchor="nw")
        ncv.configure(yscrollcommand=ncs.set)
        ncv.pack(side="left",fill="both",expand=True); ncs.pack(side="right",fill="y")

        cmds=self.nmap_cmds.get(str(port),[])
        if cmds:
            for cmd in cmds:
                nr=tk.Frame(nci,bg=CARD_BG,pady=7,padx=14); nr.pack(fill="x",pady=3)
                tk.Label(nr,text=cmd,fg=KNOCK_ACC,bg=CARD_BG,
                         font=("Consolas",9),anchor="w",wraplength=600).pack(side="left",fill="x",expand=True)
                tk.Button(nr,text="Copy",bg=CARD_BG,fg=TEXT_DIM,
                          font=("Segoe UI",8),bd=1,relief="solid",padx=8,pady=3,cursor="hand2",
                          command=lambda c=cmd:self._copy(c)).pack(side="right")
        else:
            tk.Label(nci,text="Run Automated mode scan for AI-generated Nmap commands.",
                     fg=TEXT_DIM,bg=BG,font=("Segoe UI",10)).pack(pady=20)

        ai_box=scrolledtext.ScrolledText(win,height=5,bg=SIDEBAR_BG,fg=KNOCK_ACC,
                                          font=("Consolas",8),bd=0,padx=8,pady=6)
        ai_box.pack(fill="x",padx=20,pady=(0,4))
        def ask_nmap_ai():
            ai_box.configure(state="normal"); ai_box.delete("1.0","end")
            ai_box.insert("end","Asking Knock-2 AI…\n"); ai_box.configure(state="disabled")
            def do():
                resp=gemini_ask(
                    f"Port {port} ({service}) is open on {ip}.\n"
                    "Provide the most effective Nmap commands with full flags and scripts.\n"
                    "One command per line with brief explanation.",
                    "You are an expert penetration tester.")
                ai_box.configure(state="normal")
                ai_box.insert("end",resp+"\n"); ai_box.see("end")
                ai_box.configure(state="disabled")
            threading.Thread(target=do,daemon=True).start()
        tk.Button(win,text="🤖 Ask AI for Nmap Commands",
                  bg=KNOCK_ACC,fg="#000",font=("Segoe UI",9,"bold"),
                  bd=0,padx=12,pady=4,cursor="hand2",command=ask_nmap_ai).pack(padx=20,pady=(0,8))

    def _copy(self,text):
        self.root.clipboard_clear(); self.root.clipboard_append(text)

    def _run_check(self,ip,port,label,win):
        out=win._out
        out.configure(state="normal")
        out.insert("end",f"\n▶ {label}  on  {ip}:{port}\n{'─'*50}\n")
        out.configure(state="disabled")
        def do():
            lines=[]
            try:
                ok=tcp_open(ip,port,2); lines.append(f"Reachable : {'YES ✓' if ok else 'NO ✗'}")
                if not ok: raise ConnectionRefusedError
                b=grab_banner(ip,port); lines.append(f"Banner    : {b or '(none)'}")
                if port==21 and FTP_OK:
                    try:
                        ftp=ftplib.FTP(); ftp.connect(ip,21,timeout=4)
                        ftp.login("anonymous","x@x.com")
                        lines.append("Anonymous FTP : ALLOWED ⚠️"); ftp.quit()
                    except ftplib.error_perm: lines.append("Anonymous FTP : BLOCKED ✓")
                elif port in(80,8080,8000) and REQUESTS_OK:
                    r=requests.get(f"http://{ip}:{port}",timeout=4,allow_redirects=False)
                    lines+=[f"HTTP Status  : {r.status_code}",
                             f"Server       : {r.headers.get('Server','?')}"]
                elif port==3306:
                    with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as s:
                        s.settimeout(3); s.connect((ip,3306)); raw=s.recv(256)
                        ver=raw[5:].split(b"\x00")[0].decode(errors="ignore")
                        lines.append(f"MySQL Version: {ver}")
                elif port==6379:
                    with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as s:
                        s.settimeout(3); s.connect((ip,6379))
                        s.sendall(b"PING\r\n"); resp=s.recv(128).decode(errors="ignore").strip()
                        lines.append("Redis: NOT REQUIRED ⚠️" if "+PONG" in resp else f"Redis: {resp}")
            except ConnectionRefusedError: lines.append("Port closed or filtered.")
            except Exception as ex: lines.append(f"Error: {ex}")
            out.configure(state="normal"); out.insert("end","\n".join(lines)+"\n")
            out.see("end"); out.configure(state="disabled")
        threading.Thread(target=do,daemon=True).start()

    # ══════════════════════════════════════════════════════════════════════
    #  AI REPORT
    # ══════════════════════════════════════════════════════════════════════
    def _ai_report(self):
        if not self.scan_results:
            messagebox.showwarning("No Data","Run a scan first."); return
        ip=self.ip_e.get() or self.domain_e.get()
        self._ai_write("🔄 Generating Gemini security report…")
        def do():
            report=self._build_report(ip)
            self.ai_text=report
            self.root.after(0,lambda:self._ai_write(report))
            if self.scan_folder: write_file(self.scan_folder,"ai_report.txt",report)
        threading.Thread(target=do,daemon=True).start()

    def _build_report(self,ip,nmap_advice="") -> str:
        stamp=datetime.datetime.now().strftime("%B %d, %Y %H:%M")
        assessor=self.assessor_var.get()
        ports="\n".join(f"• Port {r[0]}/{r[2]} ({r[1]})  Banner:{r[4]}"
                        for r in self.scan_results)
        vb=""
        for ps,d in self.vuln_results.items():
            vb+=f"\nPort {ps} ({d['service']}):\n"
            for v in d["vulnerabilities"]:
                vb+=(f"  [{v['severity']}] CVSS:{v.get('cvss','N/A')} "
                     f"CVE:{v['cve']} — {v['description']}\n"
                     f"  Risk: {v['risk']}\n  Fix: {v['mitigation']}\n")
        nmap_sec=f"\nNMAP AI COMMANDS:\n{nmap_advice}\n" if nmap_advice else ""
        return gemini_ask(
            f"""You are a senior cybersecurity analyst at Supraja Technologies.
Generate a PROFESSIONAL security report.

TARGET: {ip}  |  DATE: {stamp}  |  ASSESSOR: {assessor}
COMPANY: Supraja Technologies Cyber Security Cell

PORT RESULTS:
{ports}

VULNERABILITIES:
{vb or 'None detected.'}
{nmap_sec}

STRUCTURE (use these exact headings):
## 1. Executive Summary
## 2. Scan Details
## 3. Open Ports — Full Listing
## 4. Vulnerability Section
## 5. Potential Risks
## 6. Mitigations — Full List
## 7. CVSS Score Summary
## 8. Nmap Recommended Commands
## 9. Conclusion

Footer:
Assessed by: {assessor} | Supraja Technologies Cyber Security Cell
www.suprajatechnologies.com

Be professional, technical, actionable. Under 800 words.""",
            "You are a professional cybersecurity report writer for Supraja Technologies.")

    # ══════════════════════════════════════════════════════════════════════
    #  EXPORT
    # ══════════════════════════════════════════════════════════════════════
    def _export_csv(self):
        if not self.scan_results: messagebox.showwarning("No Data","No results."); return
        path=filedialog.asksaveasfilename(defaultextension=".csv",filetypes=[("CSV","*.csv")])
        if not path: return
        with open(path,"w",newline="",encoding="utf-8") as f:
            w=csv.writer(f)
            w.writerow(["Port","Service","Protocol","Status","Banner"])
            w.writerows(self.scan_results)
        messagebox.showinfo("Exported",f"Saved:\n{path}")

    def _save_pdf(self):
        if not REPORTLAB_OK: messagebox.showerror("Missing","pip install reportlab"); return
        if not self.scan_results: messagebox.showwarning("No Data","Run a scan first."); return
        path=filedialog.asksaveasfilename(defaultextension=".pdf",
                                           filetypes=[("PDF","*.pdf")],
                                           initialfile="security_report.pdf")
        if path: self._build_pdf(path,"summary")

    def _save_mit(self):
        if not REPORTLAB_OK: messagebox.showerror("Missing","pip install reportlab"); return
        if not self.scan_results: messagebox.showwarning("No Data","Run a scan first."); return
        path=filedialog.asksaveasfilename(defaultextension=".pdf",
                                           filetypes=[("PDF","*.pdf")],
                                           initialfile="mitigation_report.pdf")
        if path: self._build_pdf(path,"mitigation")

    def _build_pdf(self, path: str, mode: str, silent=False):
        try:
            doc=SimpleDocTemplate(path,pagesize=A4,
                                   leftMargin=2*cm,rightMargin=2*cm,
                                   topMargin=2*cm,bottomMargin=2*cm)
            stl=getSampleStyleSheet()
            ip=self.ip_e.get() or self.domain_e.get()
            stamp=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            assessor=self.assessor_var.get()
            T=ParagraphStyle("T",parent=stl["Title"],fontSize=18,
                              textColor=rlc.HexColor("#00ff88"),spaceAfter=4)
            H2=ParagraphStyle("H2",parent=stl["Heading2"],fontSize=13,
                               textColor=rlc.HexColor("#58a6ff"),spaceBefore=12)
            H3=ParagraphStyle("H3",parent=stl["Heading3"],fontSize=11,
                               textColor=rlc.HexColor("#e3b341"),spaceBefore=8)
            BD=ParagraphStyle("BD",parent=stl["Normal"],fontName="Helvetica-Bold")
            body=stl["Normal"]
            ttl="Security Assessment Report" if mode=="summary" else "Mitigation Report"
            elems=[
                Paragraph(f"Advanced Port Scanner — {ttl}",T),
                Paragraph("Supraja Technologies Cyber Security Cell",
                          ParagraphStyle("sub",parent=body,fontSize=9,
                                          textColor=rlc.HexColor("#8b949e"))),
                Spacer(1,6),
                Table([["Target",ip,"Date",stamp],
                       ["Assessor",assessor,"Company","Supraja Technologies"],
                       ["Open Ports",str(len(self.scan_results)),"Protocol",self.proto_var.get()]],
                      colWidths=[3.5*cm,6*cm,3*cm,6*cm]),
                Spacer(1,8),
                HRFlowable(width="100%",thickness=1.5,color=rlc.HexColor("#00ff88")),
                Spacer(1,8),
                Paragraph("1. Open Ports",H2),Spacer(1,5),
            ]
            td=[["Port","Service","Protocol","Status","Banner"]]
            for r in self.scan_results: td.append([str(x) for x in r])
            t=Table(td,colWidths=[45,75,60,50,None])
            t.setStyle(TableStyle([
                ("BACKGROUND",(0,0),(-1,0),rlc.HexColor("#00ff88")),
                ("TEXTCOLOR",(0,0),(-1,0),rlc.black),
                ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
                ("FONTSIZE",(0,0),(-1,0),9),
                ("ROWBACKGROUNDS",(0,1),(-1,-1),[rlc.HexColor("#f5f5f5"),rlc.white]),
                ("GRID",(0,0),(-1,-1),0.3,rlc.grey),
                ("FONTSIZE",(0,1),(-1,-1),8),
            ]))
            elems+=[t,Spacer(1,12)]
            sc={"Critical":rlc.HexColor("#f85149"),"High":rlc.HexColor("#d29922"),
                "Medium":rlc.HexColor("#e3b341"),"Low":rlc.HexColor("#3fb950"),
                "Info":rlc.HexColor("#58a6ff")}
            if self.vuln_results:
                elems.append(Paragraph("2. Vulnerabilities",H2)); elems.append(Spacer(1,5))
                for ps,d in self.vuln_results.items():
                    elems.append(Paragraph(f"Port {ps} — {d['service']}",H3))
                    for vuln in d["vulnerabilities"]:
                        sev=vuln["severity"]; c=sc.get(sev,rlc.grey)
                        elems+=[Paragraph(
                            f'<font color="{c.hexval()}"><b>[{sev}]</b></font> '
                            f'CVSS:{vuln.get("cvss","N/A")}  CVE:{vuln["cve"]}  '
                            f'{vuln["description"]}',body),
                                Spacer(1,3)]
                elems.append(Spacer(1,8))
                # Risks
                elems.append(Paragraph("3. Potential Risks",H2)); elems.append(Spacer(1,5))
                rd=[["Port","Service","Risk"]]
                for ps,d in self.vuln_results.items():
                    for v in d["vulnerabilities"]:
                        if v["severity"] in("Critical","High"):
                            rd.append([ps,d["service"],v["risk"][:80]])
                if len(rd)>1:
                    rt=Table(rd,colWidths=[40,70,None])
                    rt.setStyle(TableStyle([
                        ("BACKGROUND",(0,0),(-1,0),rlc.HexColor("#f85149")),
                        ("TEXTCOLOR",(0,0),(-1,0),rlc.white),
                        ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
                        ("FONTSIZE",(0,0),(-1,0),9),
                        ("ROWBACKGROUNDS",(0,1),(-1,-1),[rlc.HexColor("#fff5f5"),rlc.white]),
                        ("GRID",(0,0),(-1,-1),0.3,rlc.grey),
                        ("FONTSIZE",(0,1),(-1,-1),8),
                    ]))
                    elems+=[rt,Spacer(1,8)]
                # Mitigations
                elems.append(Paragraph("4. Mitigations",H2)); elems.append(Spacer(1,5))
                for ps,d in self.vuln_results.items():
                    for vuln in d["vulnerabilities"]:
                        sev=vuln["severity"]; c=sc.get(sev,rlc.grey)
                        elems+=[
                            Paragraph(f'Port {ps} ({d["service"]}) '
                                      f'<font color="{c.hexval()}"><b>[{sev}]</b></font>',BD),
                            Paragraph(f'Fix: {vuln["mitigation"]}',body),
                            Spacer(1,4)]
                elems.append(Spacer(1,8))
                # CVSS
                elems.append(Paragraph("5. CVSS Score Summary",H2)); elems.append(Spacer(1,5))
                cd=[["Port","Service","Severity","CVSS","CVE"]]
                for ps,d in self.vuln_results.items():
                    for vuln in d["vulnerabilities"]:
                        cd.append([ps,d["service"],vuln["severity"],
                                   vuln.get("cvss","N/A"),vuln["cve"]])
                ct=Table(cd,colWidths=[40,75,70,55,None])
                ct.setStyle(TableStyle([
                    ("BACKGROUND",(0,0),(-1,0),rlc.HexColor("#21262d")),
                    ("TEXTCOLOR",(0,0),(-1,0),rlc.white),
                    ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold"),
                    ("FONTSIZE",(0,0),(-1,0),9),
                    ("ROWBACKGROUNDS",(0,1),(-1,-1),[rlc.HexColor("#f5f5f5"),rlc.white]),
                    ("GRID",(0,0),(-1,-1),0.3,rlc.grey),
                    ("FONTSIZE",(0,1),(-1,-1),8),
                ]))
                elems+=[ct,Spacer(1,8)]
            # AI
            ai_text=self.ai_out.get("1.0","end").strip()
            if ai_text and "AI report" not in ai_text[:20]:
                elems.append(Paragraph("6. AI Security Analysis",H2)); elems.append(Spacer(1,5))
                for line in ai_text.split("\n"):
                    elems.append(Paragraph(line or "&nbsp;",body)); elems.append(Spacer(1,2))
                elems.append(Spacer(1,8))
            # Footer
            elems+=[
                HRFlowable(width="100%",thickness=1,color=rlc.HexColor("#30363d")),
                Spacer(1,5),
                Paragraph("Done By",H2),
                Paragraph(f"Assessor     : {assessor}",body),
                Paragraph("Organization : Supraja Technologies Cyber Security Cell",body),
                Paragraph("Unit         : a unit of CHSMRLSS Technologies Pvt. Ltd.",body),
                Paragraph("Website      : www.suprajatechnologies.com",body),
                Paragraph(f"Date         : {stamp}",body),
                Spacer(1,6),
                Paragraph("⚠ This report is for authorized assessment only. "
                           "Unauthorized scanning is illegal.",
                           ParagraphStyle("d",parent=body,fontSize=7,
                                           textColor=rlc.HexColor("#8b949e"))),
            ]
            doc.build(elems)
            if not silent:
                messagebox.showinfo("PDF Saved",f"Saved:\n{path}")
        except Exception as exc:
            if not silent: messagebox.showerror("PDF Error",str(exc))


# ══════════════════════════════════════════════════════════════════════════════
#  ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════
def parse_prompt(text: str):
    t=text.strip().lower()
    tm=re.search(r"(\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}|[a-z0-9\-]+\.[a-z]{2,})",t)
    target=tm.group(1) if tm else None
    rm=re.search(r"(\d+)\s*[-–to]+\s*(\d+)",t)
    if rm: sp,ep=int(rm.group(1)),int(rm.group(2))
    else:
        sm=re.search(r"port[s]?\s+(\d+)",t)
        sp=ep=int(sm.group(1)) if sm else None
    return target,sp,ep

if __name__=="__main__":
    root=tk.Tk()
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except: pass
    AdvancedPortScanner(root)
    root.mainloop()
