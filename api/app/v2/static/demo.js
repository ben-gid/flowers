const MAX_BYTES = 5 * 1024 * 1024; // mirrors utils/dependencies.py
const TYPES = ["image/jpeg", "image/png", "image/jpg"];

/** Client-side pre-check. The API is still the authority. */
function fileProblem(file) {
  if (!TYPES.includes(file.type)) return "Use a JPEG or PNG image.";
  if (file.size > MAX_BYTES) return "That image is over 5 MB. Pick a smaller file.";
  return null;
}

async function classifyFile(file) {
  const body = new FormData();
  body.append("file", file);
  const res = await fetch("/classify", { method: "POST", body });
  const data = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(data.detail || `Request failed (${res.status})`);
  return data;
}

async function getHealth() {
  const res = await fetch("/health");
  if (!res.ok) throw new Error(`Health check failed (${res.status})`);
  return res.json();
}

/** Fills [data-model-name] text and [data-hf-link] hrefs from /health. */
function applyHealth(health) {
  document.querySelectorAll("[data-model-name]").forEach((el) => {
    el.textContent = health.model_name || "no model loaded";
  });
  if (!health.model_repo) return;
  document.querySelectorAll("[data-hf-link]").forEach((el) => {
    el.href = `https://huggingface.co/${health.model_repo}`;
  });
}

/**
 * Click, keyboard and drag-and-drop (anywhere in the window) onto one zone.
 * onPick(file) fires for a valid-looking file, onReject(message) otherwise.
 */
function wirePicker({ zone, input, onPick, onReject, onDragChange }) {
  const take = (files) => {
    const file = files && files[0];
    if (!file) return;
    const problem = fileProblem(file);
    problem ? onReject(problem) : onPick(file);
  };

  zone.addEventListener("click", () => input.click());
  zone.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      input.click();
    }
  });
  input.addEventListener("change", () => {
    take(input.files);
    input.value = ""; // so the same file can be picked twice
  });

  let depth = 0;
  const setDragging = (on) => onDragChange && onDragChange(on);
  window.addEventListener("dragenter", (event) => {
    event.preventDefault();
    if (++depth === 1) setDragging(true);
  });
  window.addEventListener("dragover", (event) => event.preventDefault());
  window.addEventListener("dragleave", () => {
    if (--depth <= 0) {
      depth = 0;
      setDragging(false);
    }
  });
  window.addEventListener("drop", (event) => {
    event.preventDefault();
    depth = 0;
    setDragging(false);
    take(event.dataTransfer && event.dataTransfer.files);
  });
}
