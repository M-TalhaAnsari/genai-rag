(() => {
  const state = { mode: "recipe", file: null };

  const els = {
    toggleOptions: document.querySelectorAll(".toggle__option"),
    dietaryBlock: document.getElementById("dietary-block"),
    dietaryInput: document.getElementById("dietary-input"),
    dropzone: document.getElementById("dropzone"),
    imageInput: document.getElementById("image-input"),
    previewImg: document.getElementById("preview-img"),
    dropzoneEmpty: document.getElementById("dropzone-empty"),
    analyzeBtn: document.getElementById("analyze-btn"),
    analyzeBtnLabel: document.getElementById("analyze-btn-label"),
    errorMsg: document.getElementById("error-msg"),
    stateEmpty: document.getElementById("state-empty"),
    stateLoading: document.getElementById("state-loading"),
    stateError: document.getElementById("state-error"),
    stateErrorDetail: document.getElementById("state-error-detail"),
    stateResults: document.getElementById("state-results"),
    loadingLabel: document.getElementById("loading-label"),
  };

  const LOADING_STEPS = {
    recipe: ["Reading the photo…", "Identifying ingredients…", "Drafting recipes…"],
    analysis: ["Reading the photo…", "Estimating the portion…", "Breaking down nutrients…"],
  };

  // ---------- Mode toggle ----------
  els.toggleOptions.forEach((btn) => {
    btn.addEventListener("click", () => {
      state.mode = btn.dataset.mode;
      els.toggleOptions.forEach((b) => {
        const active = b === btn;
        b.classList.toggle("is-active", active);
        b.setAttribute("aria-checked", String(active));
      });
      const isRecipe = state.mode === "recipe";
      els.dietaryBlock.style.opacity = isRecipe ? "1" : "0.45";
      els.dietaryInput.disabled = !isRecipe;
      updateAnalyzeLabel();
    });
  });

  // ---------- Upload: click + drag/drop ----------
  els.dropzone.addEventListener("click", (e) => {
    e.preventDefault();
    els.imageInput.click();
  });

  els.imageInput.addEventListener("change", () => {
    const file = els.imageInput.files[0];
    if (file) setFile(file);
  });

  ["dragover", "dragenter"].forEach((evt) =>
    els.dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      els.dropzone.classList.add("is-drag");
    })
  );
  ["dragleave", "drop"].forEach((evt) =>
    els.dropzone.addEventListener(evt, (e) => {
      e.preventDefault();
      els.dropzone.classList.remove("is-drag");
    })
  );
  els.dropzone.addEventListener("drop", (e) => {
    const file = e.dataTransfer.files[0];
    if (file) setFile(file);
  });

  function setFile(file) {
    if (!file.type.startsWith("image/")) {
      showError("That file isn't an image. Try a JPEG, PNG, or WEBP.");
      return;
    }
    state.file = file;
    hideError();
    const url = URL.createObjectURL(file);
    els.previewImg.src = url;
    els.previewImg.hidden = false;
    els.dropzoneEmpty.hidden = true;
    updateAnalyzeLabel();
  }

  function updateAnalyzeLabel() {
    if (!state.file) {
      els.analyzeBtn.disabled = true;
      els.analyzeBtnLabel.textContent = "Upload a photo to begin";
      return;
    }
    els.analyzeBtn.disabled = false;
    els.analyzeBtnLabel.textContent =
      state.mode === "recipe" ? "Find recipes" : "Analyze nutrition";
  }

  // ---------- Submit ----------
  els.analyzeBtn.addEventListener("click", runAnalysis);

  async function runAnalysis() {
    if (!state.file) return;
    hideError();
    setView("loading");
    cycleLoadingLabels();

    const formData = new FormData();
    formData.append("image", state.file);
    formData.append("dietary_restrictions", state.mode === "recipe" ? els.dietaryInput.value : "");
    formData.append("workflow_type", state.mode);

    try {
      const res = await fetch("/api/analyze", { method: "POST", body: formData });
      const body = await res.json().catch(() => null);

      if (!res.ok) {
        const detail = (body && body.detail) || `Request failed (${res.status})`;
        throw new Error(detail);
      }

      renderResult(body.workflow_type, body.result);
      setView("results");
    } catch (err) {
      els.stateErrorDetail.textContent = err.message || "Something went wrong. Try again.";
      setView("error");
    }
  }

  function cycleLoadingLabels() {
    const steps = LOADING_STEPS[state.mode];
    let i = 0;
    els.loadingLabel.textContent = steps[0];
    const interval = setInterval(() => {
      i += 1;
      if (i >= steps.length || els.stateLoading.hidden) {
        clearInterval(interval);
        return;
      }
      els.loadingLabel.textContent = steps[i];
    }, 1400);
  }

  function setView(view) {
    els.stateEmpty.hidden = view !== "empty";
    els.stateLoading.hidden = view !== "loading";
    els.stateError.hidden = view !== "error";
    els.stateResults.hidden = view !== "results";
  }

  function showError(msg) {
    els.errorMsg.textContent = msg;
    els.errorMsg.hidden = false;
  }
  function hideError() {
    els.errorMsg.hidden = true;
  }

  // ---------- Rendering ----------
  function renderResult(workflowType, result) {
    els.stateResults.innerHTML = "";
    if (workflowType === "recipe") {
      els.stateResults.appendChild(renderRecipes(result));
    } else {
      els.stateResults.appendChild(renderAnalysis(result));
    }
  }

  function extractRecipes(result) {
    if (result && Array.isArray(result.recipes)) return result.recipes;
    const task = result && result.recipe_suggestion_task;
    if (task && task.json_dict && Array.isArray(task.json_dict.recipes)) {
      return task.json_dict.recipes;
    }
    return [];
  }

  function renderRecipes(result) {
    const recipes = extractRecipes(result);
    const wrap = document.createElement("div");

    if (!recipes.length) {
      wrap.innerHTML = `<p class="state-empty__sub">No recipes came back for this photo. Try another one.</p>`;
      return wrap;
    }

    const list = document.createElement("div");
    list.className = "recipe-list";

    recipes.forEach((recipe, idx) => {
      const card = document.createElement("article");
      card.className = "recipe-card";

      const ingredientChips = (recipe.ingredients || [])
        .map((ing) => `<span class="ingredient-chip">${escapeHtml(ing)}</span>`)
        .join("");

      card.innerHTML = `
        <div class="recipe-card__head">
          <span class="recipe-card__index">${String(idx + 1).padStart(2, "0")}</span>
          <h3 class="recipe-card__title">${escapeHtml(recipe.title || "Untitled recipe")}</h3>
        </div>
        <div class="recipe-card__ingredients">${ingredientChips}</div>
        <p class="recipe-card__instructions">${escapeHtml(recipe.instructions || "")}</p>
        ${
          recipe.calorie_estimate
            ? `<span class="recipe-card__calories">${escapeHtml(String(recipe.calorie_estimate))} kcal</span>`
            : ""
        }
      `;
      list.appendChild(card);
    });

    wrap.appendChild(list);
    return wrap;
  }

  function renderAnalysis(result) {
    result = result || {};
    const nutrients = result.nutrients || {};
    const label = document.createElement("div");
    label.className = "nutrition-label";

    let html = `<h3 class="nutrition-label__title">Nutrition Facts</h3>`;

    if (result.dish || result.portion_size) {
      html += `<div class="nutrition-label__dish">`;
      if (result.dish) html += `<strong>${escapeHtml(result.dish)}</strong>`;
      if (result.portion_size) html += ` — ${escapeHtml(result.portion_size)}`;
      html += `</div>`;
    }

    const calories = result.total_calories ?? result.estimated_calories;
    if (calories !== undefined) {
      html += `
        <div class="nutrition-label__hero">
          <span class="nutrition-label__hero-label">Calories</span>
          <span class="nutrition-label__hero-value">${escapeHtml(String(calories))}</span>
        </div>`;
    }

    ["protein", "carbohydrates", "fats"].forEach((key) => {
      if (nutrients[key] !== undefined) {
        html += `
          <div class="nutrition-label__row">
            <strong>${capitalize(key)}</strong>
            <span>${escapeHtml(String(nutrients[key]))}</span>
          </div>`;
      }
    });

    if (Array.isArray(nutrients.vitamins) && nutrients.vitamins.length) {
      html += `<div class="nutrition-label__group-title">Vitamins</div>`;
      nutrients.vitamins.forEach((v) => {
        html += `
          <div class="nutrition-label__row">
            <span>${escapeHtml(v.name || "—")}</span>
            <span>${escapeHtml(String(v.percentage_dv ?? "—"))}</span>
          </div>`;
      });
    }

    if (Array.isArray(nutrients.minerals) && nutrients.minerals.length) {
      html += `<div class="nutrition-label__group-title">Minerals</div>`;
      nutrients.minerals.forEach((m) => {
        html += `
          <div class="nutrition-label__row">
            <span>${escapeHtml(m.name || "—")}</span>
            <span>${escapeHtml(String(m.amount ?? "—"))}</span>
          </div>`;
      });
    }

    if (result.health_evaluation) {
      html += `
        <div class="nutrition-label__eval">
          <span class="nutrition-label__eval-title">Health evaluation</span>
          ${escapeHtml(result.health_evaluation)}
        </div>`;
    }

    label.innerHTML = html;
    return label;
  }

  function capitalize(s) {
    return s.charAt(0).toUpperCase() + s.slice(1);
  }

  function escapeHtml(str) {
    const div = document.createElement("div");
    div.textContent = str;
    return div.innerHTML;
  }
})();