// ===== GSAP SCROLL-TRIGGERED ANIMATIONS =====
// Requires: GSAP 3.12+ and ScrollTrigger plugin

gsap.registerPlugin(ScrollTrigger);

// ===== SECTION TITLES — SCROLL REVEAL =====
gsap.utils.toArray(".section-title").forEach(title => {
  gsap.from(title, {
    scrollTrigger: {
      trigger: title,
      start: "top 85%",
      end: "top 40%",
      toggleActions: "play none none none"
    },
    opacity: 0,
    y: 30,
    duration: 0.8,
    ease: "power2.out"
  });
});

gsap.utils.toArray(".section-label").forEach(label => {
  gsap.from(label, {
    scrollTrigger: {
      trigger: label,
      start: "top 85%",
      toggleActions: "play none none none"
    },
    opacity: 0,
    y: 20,
    duration: 0.6,
    ease: "power2.out"
  });
});

gsap.utils.toArray(".section-intro").forEach(intro => {
  gsap.from(intro, {
    scrollTrigger: {
      trigger: intro,
      start: "top 85%",
      toggleActions: "play none none none"
    },
    opacity: 0,
    y: 20,
    duration: 0.8,
    delay: 0.15,
    ease: "power2.out"
  });
});

// ===== TIMELINE ITEMS — STAGGERED REVEAL =====
gsap.utils.toArray(".timeline-item").forEach((item, index) => {
  gsap.to(item, {
    scrollTrigger: {
      trigger: item,
      start: "top 88%",
      toggleActions: "play none none none"
    },
    opacity: 1,
    y: 0,
    duration: 0.7,
    delay: index * 0.05,
    ease: "power2.out"
  });
});

// ===== WHY CARDS — STAGGER =====
gsap.utils.toArray(".why-card").forEach((card, index) => {
  gsap.from(card, {
    scrollTrigger: {
      trigger: card,
      start: "top 88%",
      toggleActions: "play none none none"
    },
    opacity: 0,
    y: 30,
    scale: 0.95,
    duration: 0.6,
    delay: index * 0.1,
    ease: "power2.out"
  });
});

// ===== PROJECT CARDS — SCROLL REVEAL =====
gsap.utils.toArray(".project-card").forEach((card, index) => {
  gsap.from(card, {
    scrollTrigger: {
      trigger: card,
      start: "top 90%",
      toggleActions: "play none none none"
    },
    opacity: 0,
    y: 40,
    duration: 0.7,
    delay: index * 0.08,
    ease: "power2.out"
  });
});

// ===== STAT CARDS — STAGGERED FLOAT IN =====
gsap.utils.toArray(".stat-card").forEach((card, index) => {
  gsap.from(card, {
    scrollTrigger: {
      trigger: card,
      start: "top 90%",
      toggleActions: "play none none none"
    },
    opacity: 0,
    x: 30,
    duration: 0.6,
    delay: index * 0.15,
    ease: "power2.out"
  });
});

// ===== CV EXPERIENCE ITEMS — REVEAL =====
gsap.utils.toArray(".cv-exp-item").forEach((item, index) => {
  gsap.from(item, {
    scrollTrigger: {
      trigger: item,
      start: "top 88%",
      toggleActions: "play none none none"
    },
    opacity: 0,
    x: -20,
    duration: 0.6,
    delay: index * 0.06,
    ease: "power2.out"
  });
});

// ===== SKILL ITEMS — STAGGER =====
gsap.utils.toArray(".skill-item").forEach((item, index) => {
  gsap.from(item, {
    scrollTrigger: {
      trigger: item,
      start: "top 92%",
      toggleActions: "play none none none"
    },
    opacity: 0,
    x: -10,
    duration: 0.4,
    delay: index * 0.04,
    ease: "power2.out"
  });
});

// ===== EDGE ITEMS — SCALE IN =====
gsap.utils.toArray(".edge-item").forEach((item, index) => {
  gsap.from(item, {
    scrollTrigger: {
      trigger: item,
      start: "top 88%",
      toggleActions: "play none none none"
    },
    opacity: 0,
    scale: 0.9,
    duration: 0.5,
    delay: index * 0.1,
    ease: "back.out(1.4)"
  });
});

// ===== PILLAR ITEMS — STAGGER =====
gsap.utils.toArray(".pillar").forEach((pillar, index) => {
  gsap.from(pillar, {
    scrollTrigger: {
      trigger: pillar,
      start: "top 88%",
      toggleActions: "play none none none"
    },
    opacity: 0,
    y: 20,
    duration: 0.5,
    delay: index * 0.08,
    ease: "power2.out"
  });
});

// ===== CONTACT SECTION — REVEAL =====
gsap.from(".contact-quote", {
  scrollTrigger: {
    trigger: ".contact-section",
    start: "top 75%",
    toggleActions: "play none none none"
  },
  opacity: 0,
  y: 30,
  duration: 0.8,
  ease: "power2.out"
});

gsap.from(".contact-sub", {
  scrollTrigger: {
    trigger: ".contact-section",
    start: "top 75%",
    toggleActions: "play none none none"
  },
  opacity: 0,
  y: 20,
  duration: 0.7,
  delay: 0.2,
  ease: "power2.out"
});

gsap.from(".contact-links", {
  scrollTrigger: {
    trigger: ".contact-section",
    start: "top 70%",
    toggleActions: "play none none none"
  },
  opacity: 0,
  y: 20,
  duration: 0.7,
  delay: 0.35,
  ease: "power2.out"
});

// ===== DIVIDERS — SCALE IN =====
gsap.utils.toArray(".divider").forEach(div => {
  gsap.from(div, {
    scrollTrigger: {
      trigger: div,
      start: "top 90%",
      toggleActions: "play none none none"
    },
    scaleX: 0,
    duration: 0.8,
    ease: "power2.inOut"
  });
});

// ===== SOCIAL CARDS — STAGGER =====
gsap.utils.toArray(".social-card").forEach((card, index) => {
  gsap.from(card, {
    scrollTrigger: {
      trigger: card,
      start: "top 88%",
      toggleActions: "play none none none"
    },
    opacity: 0,
    y: 25,
    duration: 0.5,
    delay: index * 0.1,
    ease: "power2.out"
  });
});

// ===== CV SKILLS BLOCKS — STAGGER =====
gsap.utils.toArray(".cv-skills-block").forEach((block, index) => {
  gsap.from(block, {
    scrollTrigger: {
      trigger: block,
      start: "top 88%",
      toggleActions: "play none none none"
    },
    opacity: 0,
    y: 20,
    duration: 0.6,
    delay: index * 0.1,
    ease: "power2.out"
  });
});

// ===== FILTER BUTTONS — STAGGER =====
gsap.utils.toArray(".filter-btn").forEach((btn, index) => {
  gsap.from(btn, {
    scrollTrigger: {
      trigger: ".project-filters",
      start: "top 88%",
      toggleActions: "play none none none"
    },
    opacity: 0,
    y: 10,
    duration: 0.3,
    delay: index * 0.05,
    ease: "power2.out"
  });
});

// ===== RESPONSIVE ADJUSTMENTS =====
ScrollTrigger.matchMedia({
  "(max-width: 640px)": function() {
    // Reduce animation intensity on small screens
    gsap.defaults({ duration: 0.5 });
  }
});

// ===== REFRESH ON RESIZE =====
let resizeTimeout;
window.addEventListener("resize", () => {
  clearTimeout(resizeTimeout);
  resizeTimeout = setTimeout(() => ScrollTrigger.refresh(), 250);
});
