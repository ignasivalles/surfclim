// nav.js — Navbar scroll shrink + mobile toggle
(function () {
  'use strict';

  const navbar   = document.getElementById('navbar');
  const toggle   = document.getElementById('navToggle');
  const navLinks = document.getElementById('navLinks');

  // ── Navbar shrink on scroll ──
  window.addEventListener('scroll', function () {
    navbar.classList.toggle('scrolled', window.scrollY > 60);
  }, { passive: true });

  // ── Mobile hamburger toggle ──
  toggle.addEventListener('click', function () {
    navLinks.classList.toggle('open');
  });

  navLinks.addEventListener('click', function (e) {
    if (e.target.tagName === 'A') {
      navLinks.classList.remove('open');
    }
  });

})();
