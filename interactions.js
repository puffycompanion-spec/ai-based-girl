// ===== LANGUAGE TOGGLE =====
function setLang(lang) {
  document.body.className = "lang-" + lang;
  document.querySelectorAll(".lang-btn").forEach(b => b.classList.remove("active"));
  const btn = document.querySelector(`.lang-btn[onclick="setLang('${lang}')"]`);
  if (btn) btn.classList.add("active");
  // Save preference
  localStorage.setItem("portfolio-lang", lang);
  // Update html lang attribute
  document.documentElement.lang = lang === "tr" ? "tr" : "en";
}

// ===== RESTORE SAVED LANGUAGE =====
(function() {
  const saved = localStorage.getItem("portfolio-lang");
  if (saved) setLang(saved);
})();

// ===== PROJECT FILTER =====
function filterProjects(cat, btn) {
  document.querySelectorAll(".filter-btn").forEach(b => b.classList.remove("active"));
  btn.classList.add("active");
  document.querySelectorAll(".project-card").forEach(card => {
    const cats = card.dataset.category || "";
    if (cat === "all" || cats.includes(cat)) {
      card.style.display = "flex";
      card.style.flexDirection = "column";
    } else {
      card.style.display = "none";
    }
  });
}

// ===== NAVBAR SCROLL EFFECT =====
const navbar = document.querySelector(".lang-bar");
let lastScrollY = 0;

window.addEventListener("scroll", () => {
  const scrollY = window.pageYOffset || document.documentElement.scrollTop;

  // Scrolled state — add shadow
  if (scrollY > 60) {
    navbar.classList.add("scrolled");
  } else {
    navbar.classList.remove("scrolled");
  }

  // Active section spy
  const sections = document.querySelectorAll("section[id]");
  const navLinks = document.querySelectorAll(".nav-links a");
  let current = "";
  sections.forEach(s => {
    if (scrollY >= s.offsetTop - 200) current = s.getAttribute("id");
  });
  navLinks.forEach(a => {
    a.classList.remove("active");
    if (a.getAttribute("href") === "#" + current) a.classList.add("active");
  });

  lastScrollY = scrollY;
});

// ===== MOBILE HAMBURGER MENU =====
const hamburger = document.querySelector(".nav-hamburger");
const drawer = document.querySelector(".mobile-drawer");
const overlay = document.querySelector(".drawer-overlay");

if (hamburger && drawer && overlay) {
  hamburger.addEventListener("click", () => {
    hamburger.classList.toggle("open");
    drawer.classList.toggle("open");
    overlay.classList.toggle("open");
    document.body.style.overflow = drawer.classList.contains("open") ? "hidden" : "";
  });

  overlay.addEventListener("click", () => {
    hamburger.classList.remove("open");
    drawer.classList.remove("open");
    overlay.classList.remove("open");
    document.body.style.overflow = "";
  });

  drawer.querySelectorAll("a").forEach(link => {
    link.addEventListener("click", () => {
      hamburger.classList.remove("open");
      drawer.classList.remove("open");
      overlay.classList.remove("open");
      document.body.style.overflow = "";
    });
  });
}

// ===== CONSOLE EASTER EGG =====
console.log(
  "%c✨ Fatma Zehra Dinç — AI Engineer & Product Builder ✨",
  "font-size: 16px; color: #C4714A; font-weight: bold;"
);
