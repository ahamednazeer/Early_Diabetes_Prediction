document.addEventListener("DOMContentLoaded", () => {
  const forms = document.querySelectorAll("form");
  forms.forEach((form) => {
    form.addEventListener("submit", () => {
      const button = form.querySelector('button[type="submit"]');
      if (!button) return;
      button.disabled = true;
      button.dataset.original = button.textContent;
      button.textContent = "Processing...";
      setTimeout(() => {
        button.disabled = false;
        if (button.dataset.original) {
          button.textContent = button.dataset.original;
        }
      }, 3000);
    });
  });
});
