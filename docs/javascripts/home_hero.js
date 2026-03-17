let currentHero = null;

function updateHomeHero() {
  if (!currentHero) {
    return;
  }

  const rect = currentHero.getBoundingClientRect();
  const range = Math.max(currentHero.offsetHeight * 0.9, 1);
  const progress = Math.min(Math.max(-rect.top / range, 0), 1);
  const opacity = 1 - progress * 0.78;
  const shift = `${progress * -32}px`;
  currentHero.style.setProperty("--hero-opacity", opacity.toFixed(3));
  currentHero.style.setProperty("--hero-shift", shift);
}

function wireHomeHero() {
  currentHero = document.querySelector("[data-home-hero]");
  updateHomeHero();
}

if (!window.__fusdbHomeHeroBound) {
  window.__fusdbHomeHeroBound = true;
  window.addEventListener("scroll", updateHomeHero, { passive: true });
  window.addEventListener("resize", updateHomeHero, { passive: true });
}

if (typeof document$ !== "undefined" && document$.subscribe) {
  document$.subscribe(wireHomeHero);
} else {
  document.addEventListener("DOMContentLoaded", wireHomeHero);
}
