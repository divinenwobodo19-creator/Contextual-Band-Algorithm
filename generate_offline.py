#!/usr/bin/env python3
"""
Generate a standalone offline teacher HTML file.
No Python or server needed to use the output — just open in any browser.

Usage:  PYTHONPATH=. python3 generate_offline.py
        → creates offline_teacher.html (self-contained, ~50KB)
"""

import json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from linucb_brain import Brain

BRAIN_FILE = "brain_state.json"
OUTPUT = "offline_teacher.html"

if not os.path.exists(BRAIN_FILE):
    print(f"No {BRAIN_FILE} found. Run demo_school.py or teacher_portal.py first.")
    sys.exit(1)

brain = Brain.load(BRAIN_FILE)
data = brain.export_teacher_data()

HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Teacher Offline Tool</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: -apple-system, Arial, sans-serif; background: #f5f6fa; color: #2c3e50; padding: 20px; }
  .container { max-width: 1000px; margin: auto; }
  h1 { color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; margin-bottom: 20px; }
  h2 { margin: 20px 0 10px; color: #34495e; }
  .tabs { display: flex; gap: 4px; margin-bottom: 20px; flex-wrap: wrap; }
  .tab { padding: 10px 20px; background: #ddd; border: none; cursor: pointer; border-radius: 6px 6px 0 0; font-size: 14px; font-weight: 600; }
  .tab.active { background: #3498db; color: #fff; }
  .tab-content { display: none; background: #fff; padding: 20px; border-radius: 0 6px 6px 6px; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }
  .tab-content.active { display: block; }
  label { display: block; margin: 8px 0 2px; font-weight: 600; }
  input[type=text], input[type=number] { width: 100%; padding: 8px; margin-bottom: 8px; border: 1px solid #ccc; border-radius: 4px; font-size: 14px; }
  .score-row { display: flex; align-items: center; gap: 10px; margin: 2px 0; }
  .score-row label { flex: 1; margin: 0; font-weight: normal; }
  .score-row input { flex: 1; max-width: 100px; }
  .tier-box { border-radius: 8px; padding: 15px; margin: 10px 0; }
  .tier-remediation { background: #fdecec; border-left: 5px solid #e74c3c; }
  .tier-on_track { background: #fef9e7; border-left: 5px solid #f1c40f; }
  .tier-ahead { background: #e8f8f5; border-left: 5px solid #27ae60; }
  .tier-box h3 { margin-bottom: 4px; }
  .tier-note { font-size: 13px; color: #555; font-style: italic; margin-bottom: 8px; }
  .student-list { margin: 6px 0; }
  .student-list li { margin: 4px 0; list-style: none; }
  .btn { background: #3498db; color: #fff; border: none; padding: 12px 24px; border-radius: 6px; font-size: 15px; font-weight: 600; cursor: pointer; width: 100%; }
  .btn:hover { background: #2980b9; }
  .btn-green { background: #27ae60; }
  .btn-green:hover { background: #219a52; }
  .stats { display: flex; gap: 15px; flex-wrap: wrap; margin-bottom: 20px; }
  .stat-card { background: #fff; border: 1px solid #ddd; border-radius: 8px; padding: 15px 20px; flex: 1; min-width: 120px; text-align: center; }
  .stat-card .num { font-size: 28px; font-weight: 700; color: #3498db; }
  .stat-card .label { font-size: 12px; color: #888; margin-top: 4px; }
  .print-area { padding: 10px; }
  .msg { margin-top: 8px; font-weight: 600; }
  .msg-success { color: #27ae60; }
  @media print { body { background: #fff; padding: 0; } .tab { display: none; } .tab-content { display: block !important; } .btn { display: none; } }
</style>
</head>
<body>
<div class="container">
<h1>Teacher Offline Tool</h1>
<p id="status">Loading data...</p>

<div class="tabs" id="tabs">
  <button class="tab active" onclick="switchTab('input')">Enter Scores</button>
  <button class="tab" onclick="switchTab('triage')">Class Groups</button>
  <button class="tab" onclick="switchTab('report')">Print Report</button>
</div>

<div id="tab-input" class="tab-content active">
  <h2>What subject are you teaching?</h2>
  <input type="text" id="subject-input" placeholder="e.g. Algebra, Fractions, Cell Biology" oninput="renderInputForm()">
  <div id="score-form"></div>
  <button class="btn btn-green" onclick="submitScores()">Submit Scores</button>
  <p id="submit-msg" class="msg"></p>
  <hr style="margin:20px 0;">
  <details style="cursor:pointer;">
    <summary style="font-weight:600;color:#34495e;">Edit student names</summary>
    <div id="name-editor" style="margin-top:10px;"></div>
  </details>
</div>

<div id="tab-triage" class="tab-content">
  <h2>Class Groups</h2>
  <select id="triage-subject" onchange="runTriage()"></select>
  <div id="triage-results"></div>
</div>

<div id="tab-report" class="tab-content">
  <h2>Printable Report</h2>
  <select id="report-subject" onchange="generateReport()"></select>
  <div id="report-results" class="print-area"></div>
  <button class="btn" onclick="window.print()" style="margin-top:15px;">Print Report</button>
</div>
</div>

<script>
// ======================== EMBEDDED DATA ========================
var DATA = __DATA__;

var TIERS = {
  remediation: { label: "Needs Extra Help", icon: "🔴",
    note: "Scored below 40% (F9). Needs extra practice — give foundational materials.",
    color: "#e74c3c" },
  on_track: { label: "At Expected Level", icon: "🟡",
    note: "Scored 40–74% (E8 to B2). On track — continue with standard curriculum.",
    color: "#f1c40f" },
  ahead: { label: "Ahead of Class", icon: "🟢",
    note: "Scored 75% or above (A1). Ahead of class — give advanced materials.",
    color: "#27ae60" }
};

// ======================== HELPERS ========================
function nigerianGrade(score) {
  var p = score * 100;
  if (p >= 75) return "A1";
  if (p >= 70) return "B2";
  if (p >= 65) return "B3";
  if (p >= 60) return "C4";
  if (p >= 55) return "C5";
  if (p >= 50) return "C6";
  if (p >= 45) return "D7";
  if (p >= 40) return "E8";
  return "F9";
}

function predictGrade(studentId, subject) {
  var s = DATA.students.find(function(st) { return st.id === studentId; });
  if (!s) return 0.5;
  var grades = s.grade_history[subject];
  if (grades && grades.length > 0) {
    var sum = grades.reduce(function(a, b) { return a + b; }, 0);
    return Math.min(Math.max(sum / grades.length, 0), 1);
  }
  return Math.min(Math.max(s.performance, 0), 1);
}

function triage(subject) {
  var results = DATA.students.map(function(s) {
    var pred = predictGrade(s.id, subject);
    return { student_id: s.id, name: s.name, predicted: pred };
  });
  var tiers = { remediation: [], on_track: [], ahead: [] };
  results.forEach(function(r) {
    var t;
    if (r.predicted < 0.4) t = 'remediation';
    else if (r.predicted < 0.75) t = 'on_track';
    else t = 'ahead';
    tiers[t].push(r);
  });
  return { subject: subject, total: results.length, tiers: tiers };
}

// ======================== TABS ========================
function switchTab(name) {
  document.querySelectorAll('.tab-content').forEach(function(el) { el.classList.remove('active'); });
  document.querySelectorAll('.tab').forEach(function(el) { el.classList.remove('active'); });
  document.getElementById('tab-' + name).classList.add('active');
  var btns = document.querySelectorAll('.tab');
  var idx = ['input','triage','report'].indexOf(name);
  if (btns[idx]) btns[idx].classList.add('active');
}

// ======================== INPUT TAB ========================
function populateSubjectSelects() {
  var subs = DATA.subjects;
  ['triage-subject', 'report-subject'].forEach(function(id) {
    var sel = document.getElementById(id);
    if (!sel) return;
    sel.innerHTML = '';
    subs.forEach(function(s) {
      var opt = document.createElement('option');
      opt.value = s; opt.textContent = s;
      sel.appendChild(opt);
    });
  });
}

function renderInputForm() {
  var subject = document.getElementById('subject-input').value.trim();
  var form = document.getElementById('score-form');
  if (!subject) {
    form.innerHTML = '<p style="color:#888;margin-top:8px;">Type a subject name above to begin.</p>';
    return;
  }
  var heading = document.createElement('h3');
  heading.textContent = 'Enter scores for "' + subject + '" (0–100)';
  form.innerHTML = '';
  form.appendChild(heading);
  DATA.students.forEach(function(s) {
    var defaultVal = s.performance * 100;
    var latest = s.grade_history[subject];
    if (latest && latest.length > 0) defaultVal = latest[latest.length - 1] * 100;
    var div = document.createElement('div');
    div.className = 'score-row';
    div.innerHTML = '<label id="label-' + s.id + '">' + s.name + '</label>' +
      '<input type="number" min="0" max="100" step="1" value="' + Math.round(defaultVal) + '" id="score-' + s.id + '">';
    form.appendChild(div);
  });
}

function renderNameEditor() {
  var container = document.getElementById('name-editor');
  container.innerHTML = '';
  DATA.students.forEach(function(s) {
    var div = document.createElement('div');
    div.style.cssText = 'display:flex;align-items:center;gap:8px;margin:4px 0;';
    div.innerHTML = '<span style="min-width:40px;color:#888;font-size:13px;">' + s.id + '</span>' +
      '<input type="text" value="' + s.name + '" id="ren-' + s.id + '" style="flex:1;padding:6px;border:1px solid #ccc;border-radius:4px;">' +
      '<button onclick="saveName(\'' + s.id + '\')" style="padding:6px 12px;background:#3498db;color:#fff;border:none;border-radius:4px;cursor:pointer;">Save</button>';
    container.appendChild(div);
  });
}

function saveName(id) {
  var inp = document.getElementById('ren-' + id);
  if (!inp || !inp.value.trim()) return;
  DATA.students.forEach(function(s) {
    if (s.id === id) { s.name = inp.value.trim(); }
  });
  var label = document.getElementById('label-' + id);
  if (label) label.textContent = inp.value.trim();
  document.getElementById('submit-msg').textContent = 'Name updated to "' + inp.value.trim() + '"!';
  document.getElementById('submit-msg').className = 'msg msg-success';
  runTriage(); generateReport();
}

function submitScores() {
  var subject = document.getElementById('subject-input').value.trim();
  if (!subject) {
    document.getElementById('submit-msg').textContent = 'Please type a subject name.';
    document.getElementById('submit-msg').className = 'msg';
    return;
  }
  DATA.students.forEach(function(s) {
    var inp = document.getElementById('score-' + s.id);
    if (!inp) return;
    var score = parseFloat(inp.value) / 100;
    if (isNaN(score)) score = 0.5;
    if (!s.grade_history[subject]) s.grade_history[subject] = [];
    s.grade_history[subject].push(score);
    s.performance = (s.performance + score) / 2;
  });
  var subs = DATA.subjects;
  if (subs.indexOf(subject) === -1) { DATA.subjects.push(subject); populateSubjectSelects(); }
  var msg = document.getElementById('submit-msg');
  msg.textContent = 'Scores saved for "' + subject + '"!';
  msg.className = 'msg msg-success';
  var sel = document.getElementById('triage-subject');
  if (sel) { sel.value = subject; runTriage(); }
}

// ======================== TRIAGE TAB ========================
function runTriage() {
  var subject = document.getElementById('triage-subject').value;
  var result = triage(subject);
  var container = document.getElementById('triage-results');
  var html = '<div class="stats"><div class="stat-card"><div class="num">' + result.total + '</div><div class="label">Students Assessed</div></div></div>';
  ['remediation', 'on_track', 'ahead'].forEach(function(tierName) {
    var t = result.tiers[tierName];
    var info = TIERS[tierName];
    html += '<div class="tier-box tier-' + tierName + '">';
    html += '<h3>' + info.icon + ' ' + info.label + ' (' + t.length + ' students)</h3>';
    html += '<p class="tier-note">' + info.note + '</p>';
    if (t.length > 0) {
      html += '<ul class="student-list">';
      t.forEach(function(s) { html += '<li>- ' + s.name + ' (' + nigerianGrade(s.predicted) + ')</li>'; });
      html += '</ul>';
    } else {
      html += '<p style="color:#888;"><em>No students in this group.</em></p>';
    }
    html += '</div>';
  });
  container.innerHTML = html;
}

// ======================== REPORT TAB ========================
function generateReport() {
  var subject = document.getElementById('report-subject').value;
  var result = triage(subject);
  var now = new Date().toLocaleString();
  var html = '<h2 style="border-bottom:2px solid #3498db;padding-bottom:8px;">Class Report — ' + subject + '</h2>';
  html += '<p>Total students: ' + result.total + ' | Generated: ' + now + '</p>';
  ['remediation', 'on_track', 'ahead'].forEach(function(tierName) {
    var t = result.tiers[tierName];
    var info = TIERS[tierName];
    html += '<div class="tier-box tier-' + tierName + '">';
    html += '<h3>' + info.icon + ' ' + info.label + ' (' + t.length + ' students)</h3>';
    html += '<p class="tier-note">' + info.note + '</p>';
    if (t.length > 0) {
      html += '<ul class="student-list">';
      t.forEach(function(s) { html += '<li>- ' + s.name + ' (' + nigerianGrade(s.predicted) + ')</li>'; });
      html += '</ul>';
    } else {
      html += '<p><em>No students in this group.</em></p>';
    }
    html += '</div>';
  });
  document.getElementById('report-results').innerHTML = html;
}

// ======================== INIT ========================
document.getElementById('status').textContent = '✓ Loaded ' + DATA.students.length + ' students, ' + DATA.subjects.length + ' subjects.';
populateSubjectSelects();
renderInputForm();
renderNameEditor();
if (DATA.subjects.length > 0) { runTriage(); generateReport(); }
</script>
</body>
</html>"""

# Embed the data as JSON
html_out = HTML.replace("__DATA__", json.dumps(data))
with open(OUTPUT, "w") as f:
    f.write(html_out)

print(f"✅ Generated {OUTPUT}")
print(f"   {len(data['students'])} students, {len(data['subjects'])} subjects")
print(f"   File size: {os.path.getsize(OUTPUT) / 1024:.0f}KB")
print(f"   Open in any browser — no Python or server needed.")
