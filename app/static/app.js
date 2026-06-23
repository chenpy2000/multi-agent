const state = {
  currentRun: null,
  pollTimer: null,
  projects: [],
  selectedProject: "",
};

const els = {
  form: document.querySelector("#requestForm"),
  input: document.querySelector("#promptInput"),
  send: document.querySelector("#sendButton"),
  chatLog: document.querySelector("#chatLog"),
  workflow: document.querySelector("#workflowStrip"),
  runStatus: document.querySelector("#runStatus"),
  runSubtitle: document.querySelector("#runSubtitle"),
  runList: document.querySelector("#runList"),
  projectList: document.querySelector("#projectList"),
  newProject: document.querySelector("#newProjectButton"),
  agentGrid: document.querySelector("#agentGrid"),
  nudge: document.querySelector("#nudgeButton"),
  frameworkBadge: document.querySelector("#frameworkBadge"),
  navTabs: document.querySelectorAll(".nav-tab"),
  views: document.querySelectorAll(".view"),
};

els.navTabs.forEach((button) => {
  button.addEventListener("click", () => switchView(button.dataset.view));
});

els.form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const prompt = els.input.value.trim();
  if (!prompt) return;
  appendUserMessage(prompt);
  els.input.value = "";
  autosizeInput();
  els.send.disabled = true;

  const response = await fetch("/api/runs", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      prompt,
      selected_project: state.selectedProject || "",
    }),
  });
  const run = await response.json();
  if (!response.ok) {
    appendSystemMessage(run.error || "The request could not be started.");
    els.send.disabled = false;
    return;
  }
  setCurrentRun(run);
  startPolling(run.id);
});

els.input.addEventListener("input", autosizeInput);
els.input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    els.form.requestSubmit();
  }
});

els.newProject.addEventListener("click", () => {
  state.selectedProject = "";
  renderProjects(state.projects);
  renderProjectMode();
});

els.nudge.addEventListener("click", async () => {
  const run = state.currentRun;
  if (!run || run.agents.length < 2) return;
  const pending = run.agents.find((agent) => agent.status !== "done") || run.agents[run.agents.length - 1];
  const sender = run.agents[Math.max(0, run.agents.indexOf(pending) - 1)] || run.agents[0];
  const recipient = pending.id === sender.id ? run.agents[0] : pending;

  await fetch(`/api/runs/${run.id}/messages`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      sender_id: sender.id,
      recipient_id: recipient.id,
      content: "Please share the key constraint you need the next agent to respect.",
    }),
  });
  await refreshRun(run.id);
});

async function loadRuns() {
  const response = await fetch("/api/runs");
  const runs = await response.json();
  renderRunList(runs);
  if (!state.currentRun && runs.length) {
    setCurrentRun(runs[0]);
    if (runs[0].status !== "complete") startPolling(runs[0].id);
  }
}

function switchView(name) {
  els.navTabs.forEach((button) => button.classList.toggle("is-active", button.dataset.view === name));
  els.views.forEach((view) => view.classList.toggle("is-active", view.id === `${name}View`));
}

function startPolling(runId) {
  window.clearInterval(state.pollTimer);
  state.pollTimer = window.setInterval(async () => {
    const run = await refreshRun(runId);
    if (run && ["complete", "failed"].includes(run.status)) {
      window.clearInterval(state.pollTimer);
      els.send.disabled = false;
      await loadProjects();
    }
  }, 700);
}

async function refreshRun(runId) {
  const response = await fetch(`/api/runs/${runId}`);
  if (!response.ok) return null;
  const run = await response.json();
  setCurrentRun(run);
  return run;
}

function setCurrentRun(run) {
  state.currentRun = run;
  renderRun(run);
  loadRuns();
}

function renderRun(run) {
  renderProjectMode(run);
  els.runStatus.textContent = titleCase(run.status);
  els.runStatus.className = `status-badge ${run.status}`;
  els.frameworkBadge.textContent = `${titleCase(run.framework || "llamaindex")} runtime`;
  els.frameworkBadge.title = run.framework_note || "";
  els.nudge.disabled = !run.agents || run.agents.length < 2;
  renderWorkflow(run.agents || []);
  renderChat(run);
  renderAgents(run.agents || []);
}

async function loadProjects() {
  const response = await fetch("/api/projects");
  if (!response.ok) return;
  state.projects = await response.json();
  if (state.selectedProject && !state.projects.some((project) => project.name === state.selectedProject)) {
    state.selectedProject = "";
  }
  renderProjects(state.projects);
  renderProjectMode(state.currentRun);
}

function renderProjects(projects) {
  els.projectList.innerHTML = "";
  const newButton = document.createElement("button");
  newButton.className = `project-item ${state.selectedProject ? "" : "is-active"}`;
  newButton.type = "button";
  newButton.innerHTML = `<strong>New project</strong><span>Generate from scratch</span>`;
  newButton.addEventListener("click", () => {
    state.selectedProject = "";
    renderProjects(state.projects);
    renderProjectMode(state.currentRun);
  });
  els.projectList.appendChild(newButton);

  if (!projects.length) return;

  projects.forEach((project) => {
    const item = document.createElement("div");
    item.className = `project-item ${state.selectedProject === project.name ? "is-active" : ""}`;
    item.innerHTML = `
      <button class="project-select" type="button">
        <strong>${escapeHtml(project.name)}</strong>
        <span>${escapeHtml(project.file_count)} files</span>
      </button>
      <div class="project-actions">
        <button class="icon-button" type="button" title="Rename project" aria-label="Rename ${escapeHtml(project.name)}">&#9998;</button>
        <button class="icon-button danger" type="button" title="Delete project" aria-label="Delete ${escapeHtml(project.name)}">&#128465;</button>
      </div>
    `;
    item.querySelector(".project-select").addEventListener("click", () => {
      state.selectedProject = project.name;
      renderProjects(state.projects);
      renderProjectMode(state.currentRun);
    });
    item.querySelector(".project-actions .icon-button").addEventListener("click", () => renameProject(project.name));
    item.querySelector(".project-actions .danger").addEventListener("click", () => deleteProject(project.name));
    els.projectList.appendChild(item);
  });
}

async function renameProject(projectName) {
  const nextName = window.prompt("Rename project", projectName);
  if (!nextName || nextName.trim() === projectName) return;
  const response = await fetch(`/api/projects/${encodeURIComponent(projectName)}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ name: nextName.trim() }),
  });
  if (!response.ok) {
    const payload = await response.json();
    appendSystemMessage(payload.error || "The project could not be renamed.");
    return;
  }
  const renamed = await response.json();
  if (state.selectedProject === projectName) state.selectedProject = renamed.name;
  await loadProjects();
}

async function deleteProject(projectName) {
  if (!window.confirm(`Delete ${projectName}?`)) return;
  const response = await fetch(`/api/projects/${encodeURIComponent(projectName)}`, { method: "DELETE" });
  if (!response.ok) {
    const payload = await response.json();
    appendSystemMessage(payload.error || "The project could not be deleted.");
    return;
  }
  if (state.selectedProject === projectName) state.selectedProject = "";
  await loadProjects();
}

function renderProjectMode(run = null) {
  if (run) {
    const suffix = run.selected_project
      ? ` · Continued ${run.selected_project}`
      : state.selectedProject
        ? ` · Next: continue ${state.selectedProject}`
        : "";
    els.runSubtitle.textContent = `${run.prompt}${suffix}`;
  } else if (state.selectedProject) {
    els.runSubtitle.textContent = `Next request will continue ${state.selectedProject}`;
  } else {
    els.runSubtitle.textContent = "Describe a coding task and watch the workflow form.";
  }
}

function renderRunList(runs) {
  els.runList.innerHTML = "";
  if (!runs.length) {
    els.runList.innerHTML = `<div class="agent-meta">No runs yet</div>`;
    return;
  }
  runs.forEach((run) => {
    const button = document.createElement("button");
    button.className = `run-item ${state.currentRun?.id === run.id ? "is-active" : ""}`;
    button.type = "button";
    button.innerHTML = `<strong>${escapeHtml(run.prompt)}</strong><span>${titleCase(run.status)}</span>`;
    button.addEventListener("click", () => {
      setCurrentRun(run);
      if (run.status !== "complete") startPolling(run.id);
    });
    els.runList.appendChild(button);
  });
}

function renderWorkflow(agents) {
  els.workflow.innerHTML = "";
  const levels = workflowLevels(agents);
  levels.forEach((level, levelIndex) => {
    const lane = document.createElement("div");
    lane.className = "workflow-level";
    const label = document.createElement("div");
    label.className = "workflow-level-label";
    label.textContent = `Level ${levelIndex + 1}`;
    const nodes = document.createElement("div");
    nodes.className = "workflow-level-nodes";
    level.forEach((agent) => {
      const node = document.createElement("div");
      node.className = `workflow-node ${agent.status}`;
      const dependencies = (agent.depends_on || []).map((dependency) => `
        <span class="dependency-chip">${escapeHtml(dependency)}</span>
      `).join("");
      node.innerHTML = `
        <div class="node-dot">${levelIndex + 1}</div>
        <div class="node-body">
          <strong>${escapeHtml(agent.name)}</strong>
          <span>${escapeHtml(titleCase(agent.status))}</span>
          ${dependencies ? `<div class="dependency-list">${dependencies}</div>` : ""}
        </div>
      `;
      nodes.appendChild(node);
    });
    lane.append(label, nodes);
    els.workflow.appendChild(lane);
  });
}

function workflowLevels(agents) {
  if (!agents.length) return [];
  const groups = new Map();
  agents.forEach((agent, index) => {
    const group = Number.isInteger(agent.parallel_group) ? agent.parallel_group : index;
    if (!groups.has(group)) groups.set(group, []);
    groups.get(group).push(agent);
  });
  return Array.from(groups.entries())
    .sort(([a], [b]) => a - b)
    .map(([, items]) => items);
}

function renderChat(run) {
  els.chatLog.innerHTML = "";
  appendUserMessage(run.prompt, false);
  (run.transcript || []).forEach((message) => {
    const article = document.createElement("article");
    article.className = "message assistant";
    article.innerHTML = `
      <div class="avatar">${escapeHtml(initials(message.sender))}</div>
      <div class="bubble">
        <strong>${escapeHtml(message.sender)}</strong>
        <small> to ${escapeHtml(message.recipient)} - ${escapeHtml(message.kind)}</small>
        <p>${escapeHtml(message.content)}</p>
      </div>
    `;
    els.chatLog.appendChild(article);
  });
  if (run.project_path || (run.artifacts || []).length) {
    const article = document.createElement("article");
    article.className = "message assistant";
    const artifacts = (run.artifacts || []).map((artifact) => `
      <li><strong>${escapeHtml(artifact.name)}</strong><span>${escapeHtml(artifact.description)}</span></li>
    `).join("");
    const changes = renderChanges(run.modified_files || {});
    article.innerHTML = `
      <div class="avatar">F</div>
      <div class="bubble">
        <strong>Generated Project</strong>
        <p>${escapeHtml(run.project_path || "Files were generated.")}</p>
        ${changes}
        ${artifacts ? `<ul class="artifact-list">${artifacts}</ul>` : ""}
      </div>
    `;
    els.chatLog.appendChild(article);
  }
  els.chatLog.scrollTop = els.chatLog.scrollHeight;
}

function renderAgents(agents) {
  els.agentGrid.innerHTML = "";
  if (!agents.length) {
    els.agentGrid.innerHTML = `<div class="empty-state">No agents yet. Start a run from Chat.</div>`;
    return;
  }
  agents.forEach((agent) => {
    const card = document.createElement("article");
    card.className = "agent-card";
    const notes = (agent.messages || []).slice(-4).map((message) => `
      <div class="agent-note">
        <strong>${escapeHtml(message.sender)} to ${escapeHtml(message.recipient)}:</strong>
        ${escapeHtml(message.content)}
      </div>
    `).join("");
    card.innerHTML = `
      <header>
        <div>
          <h2>${escapeHtml(agent.name)}</h2>
          <div class="agent-meta">${escapeHtml(agent.purpose)}</div>
        </div>
        <span class="status-badge ${agent.status}">${escapeHtml(titleCase(agent.status))}</span>
      </header>
      <dl>
        <div><dt>Task</dt><dd>${escapeHtml(agent.task)}</dd></div>
        <div><dt>Input</dt><dd>${escapeHtml(agent.input)}</dd></div>
        ${agent.input_format ? `<div><dt>Input Format</dt><dd>${escapeHtml(agent.input_format)}</dd></div>` : ""}
        ${agent.expected_output_format ? `<div><dt>Expected Output</dt><dd>${escapeHtml(agent.expected_output_format)}</dd></div>` : ""}
        ${agent.logic ? `<div><dt>Logic</dt><dd>${escapeHtml(agent.logic)}</dd></div>` : ""}
        ${agent.deliverable ? `<div><dt>Deliverable</dt><dd>${escapeHtml(agent.deliverable)}</dd></div>` : ""}
        <div><dt>Dependencies</dt><dd>${escapeHtml((agent.depends_on || []).join(", ") || "None")}</dd></div>
        <div><dt>Parallel Group</dt><dd>${escapeHtml(String((agent.parallel_group ?? 0) + 1))}</dd></div>
        <div><dt>Output</dt><dd>${escapeHtml(agent.output || "Waiting for this agent to run.")}</dd></div>
      </dl>
      <div class="agent-message-list">${notes || `<div class="agent-note">No messages yet.</div>`}</div>
    `;
    els.agentGrid.appendChild(card);
  });
}

function renderChanges(changes) {
  const groups = ["created", "modified", "deleted"].map((key) => {
    const files = Array.isArray(changes[key]) ? changes[key] : [];
    if (!files.length) return "";
    const items = files.map((path) => `<li>${escapeHtml(path)}</li>`).join("");
    return `<div class="change-group"><strong>${titleCase(key)}</strong><ul>${items}</ul></div>`;
  }).join("");
  return groups ? `<div class="change-list">${groups}</div>` : "";
}

function appendUserMessage(content, scroll = true) {
  const article = document.createElement("article");
  article.className = "message user";
  article.innerHTML = `
    <div class="avatar">U</div>
    <div class="bubble"><strong>You</strong><p>${escapeHtml(content)}</p></div>
  `;
  els.chatLog.appendChild(article);
  if (scroll) els.chatLog.scrollTop = els.chatLog.scrollHeight;
}

function appendSystemMessage(content) {
  const article = document.createElement("article");
  article.className = "message assistant";
  article.innerHTML = `
    <div class="avatar">!</div>
    <div class="bubble"><strong>System</strong><p>${escapeHtml(content)}</p></div>
  `;
  els.chatLog.appendChild(article);
}

function autosizeInput() {
  els.input.style.height = "auto";
  els.input.style.height = `${Math.min(els.input.scrollHeight, 180)}px`;
}

function initials(name) {
  return name.split(/\s+/).map((part) => part[0]).join("").slice(0, 2).toUpperCase();
}

function titleCase(value) {
  return String(value || "").replace(/-/g, " ").replace(/\b\w/g, (char) => char.toUpperCase());
}

function escapeHtml(value) {
  return String(value ?? "").replace(/[&<>"']/g, (char) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#039;",
  }[char]));
}

async function loadFramework() {
  const response = await fetch("/api/framework");
  if (!response.ok) return;
  const profile = await response.json();
  els.frameworkBadge.textContent = `${titleCase(profile.active || "llamaindex")} runtime`;
  els.frameworkBadge.title = profile.rationale || "";
}

loadFramework();
loadProjects();
loadRuns();
