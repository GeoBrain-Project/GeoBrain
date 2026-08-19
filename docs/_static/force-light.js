// The GeoBrain documentation ships one palette, and the theme switcher is
// removed from the navbar. The theme still resolves its mode from
// localStorage first and from the operating system second, so a visitor
// who once chose dark on any pydata-sphinx-theme site served from the same
// origin, or who simply runs a dark desktop, would be shown a dark page
// with light-only styling on top of it and no control to undo it.
//
// This runs in the head, before the theme's deferred script, and settles
// the question in both places the theme reads.
(function () {
  var root = document.documentElement;
  try {
    window.localStorage.setItem("mode", "light");
    window.localStorage.setItem("theme", "light");
  } catch (e) {
    /* private browsing: the dataset below is still authoritative */
  }
  root.dataset.defaultMode = "light";
  root.dataset.mode = "light";
  root.dataset.theme = "light";
})();
