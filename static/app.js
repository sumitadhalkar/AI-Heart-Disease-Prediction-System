(function () {
  "use strict";

  // Everything here only runs on the result page's monitor panel, and the
  // page is already correct (final values rendered server-side) without
  // this script -- it just adds one page-load animation on top.
  var reduceMotion =
    window.matchMedia &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  if (reduceMotion) {
    return;
  }

  function animateFrom(el, prop, startValue, targetValue) {
    el.style.transition = "none";
    el.style[prop] = startValue;
    // Force a reflow so the browser applies the "none" transition before
    // we restore it -- otherwise the jump to the target would also skip
    // the CSS transition.
    el.getBoundingClientRect();
    el.style.transition = "";
    requestAnimationFrame(function () {
      el.style[prop] = targetValue;
    });
  }

  document.addEventListener("DOMContentLoaded", function () {
    document.querySelectorAll(".monitor-ring-value").forEach(function (ring) {
      var target = ring.getAttribute("data-target-offset");
      var full = ring.getAttribute("stroke-dasharray");
      if (target && full) {
        animateFrom(ring, "strokeDashoffset", full, target);
      }
    });

    document.querySelectorAll(".monitor-scale-fill").forEach(function (fill) {
      var target = fill.style.width;
      if (target) {
        animateFrom(fill, "width", "0%", target);
      }
    });

    document.querySelectorAll(".monitor-trace path").forEach(function (path) {
      path.classList.add("animate-trace");
    });
  });
})();
