(function () {
  function normalise(value) {
    return (value || "").toLowerCase().trim();
  }

  function labelFor(tag) {
    return tag
      .split("-")
      .map(function (part) {
        if (part === "mhd" || part === "hedp" || part === "iaea" || part === "ictp" || part === "ipam" || part === "mit" || part === "mipse" || part === "nndc" || part === "nist" || part === "ornl" || part === "pppl") {
          return part.toUpperCase();
        }
        return part.charAt(0).toUpperCase() + part.slice(1);
      })
      .join(" ");
  }

  function initResourceCatalog(root) {
    var catalog = root.querySelector("[data-resource-catalog]");
    if (!catalog || catalog.dataset.ready === "true") return;

    var markers = Array.prototype.slice.call(root.querySelectorAll(".resource-marker"));
    if (!markers.length) return;

    catalog.dataset.ready = "true";

    var input = catalog.querySelector(".resource-search");
    var clear = catalog.querySelector("[data-resource-clear]");
    var tagHost = catalog.querySelector("[data-resource-tags]");
    var countHost = catalog.querySelector("[data-resource-count]");
    var emptyHost = catalog.querySelector("[data-resource-empty]");
    var selectedTags = new Set();
    var allTags = new Set();

    var entries = markers.map(function (marker) {
      var item = marker.closest("li");
      var tags = normalise(marker.dataset.tags).split(/\s+/).filter(Boolean);
      tags.forEach(function (tag) { allTags.add(tag); });

      if (item) {
        item.classList.add("resource-item");
        item.dataset.resourceTags = tags.join(" ");

        var chipRow = document.createElement("span");
        chipRow.className = "resource-item-tags";
        chipRow.setAttribute("aria-label", "Resource tags");
        tags.forEach(function (tag) {
          var chip = document.createElement("button");
          chip.type = "button";
          chip.className = "resource-tag resource-tag--item";
          chip.dataset.resourceTag = tag;
          chip.textContent = labelFor(tag);
          chipRow.appendChild(chip);
        });
        item.appendChild(chipRow);
      }

      return {
        item: item,
        tags: tags,
        text: item ? normalise(item.textContent) : ""
      };
    }).filter(function (entry) { return entry.item; });

    Array.from(allTags).sort().forEach(function (tag) {
      var button = document.createElement("button");
      button.type = "button";
      button.className = "resource-tag resource-tag--filter";
      button.dataset.resourceTag = tag;
      button.setAttribute("aria-pressed", "false");
      button.textContent = labelFor(tag);
      tagHost.appendChild(button);
    });

    function syncTagButtons() {
      root.querySelectorAll("[data-resource-tag]").forEach(function (button) {
        var active = selectedTags.has(button.dataset.resourceTag);
        button.classList.toggle("is-active", active);
        if (button.classList.contains("resource-tag--filter")) {
          button.setAttribute("aria-pressed", active ? "true" : "false");
        }
      });
    }

    function applyFilters() {
      var query = normalise(input ? input.value : "");
      var shown = 0;

      entries.forEach(function (entry) {
        var matchesText = !query || entry.text.indexOf(query) !== -1 || entry.tags.some(function (tag) { return tag.indexOf(query) !== -1; });
        var matchesTags = Array.from(selectedTags).every(function (tag) { return entry.tags.indexOf(tag) !== -1; });
        var visible = matchesText && matchesTags;
        entry.item.hidden = !visible;
        if (visible) shown += 1;
      });

      if (countHost) {
        var parts = [shown + " of " + entries.length + " resources"];
        if (selectedTags.size) parts.push(selectedTags.size + " tag" + (selectedTags.size === 1 ? "" : "s") + " selected");
        countHost.textContent = parts.join(" · ");
      }
      if (emptyHost) emptyHost.hidden = shown !== 0;
      syncTagButtons();
    }

    function toggleTag(tag) {
      if (!tag) return;
      if (selectedTags.has(tag)) selectedTags.delete(tag);
      else selectedTags.add(tag);
      applyFilters();
    }

    if (input) input.addEventListener("input", applyFilters);
    if (clear) {
      clear.addEventListener("click", function () {
        selectedTags.clear();
        if (input) input.value = "";
        applyFilters();
        if (input) input.focus();
      });
    }

    root.addEventListener("click", function (event) {
      var button = event.target.closest("[data-resource-tag]");
      if (!button || !root.contains(button)) return;
      toggleTag(button.dataset.resourceTag);
    });

    applyFilters();
  }

  function boot() {
    initResourceCatalog(document);
  }

  if (typeof document$ !== "undefined" && document$ && typeof document$.subscribe === "function") {
    document$.subscribe(boot);
  } else if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
