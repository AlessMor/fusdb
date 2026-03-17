function wireHeaderHomeLink() {
  const logo = document.querySelector(".md-header__button.md-logo");
  const topic = document.querySelector(
    '[data-md-component="header-title"] .md-header__topic:first-child'
  );
  if (!logo || !topic || topic.dataset.homeLinkBound === "true") {
    return;
  }

  const goHome = () => {
    window.location.href = logo.getAttribute("href") || ".";
  };

  topic.dataset.homeLinkBound = "true";
  topic.style.cursor = "pointer";
  topic.setAttribute("role", "link");
  topic.setAttribute("tabindex", "0");
  topic.addEventListener("click", goHome);
  topic.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") {
      event.preventDefault();
      goHome();
    }
  });
}

if (typeof document$ !== "undefined" && document$.subscribe) {
  document$.subscribe(wireHeaderHomeLink);
} else {
  document.addEventListener("DOMContentLoaded", wireHeaderHomeLink);
}
