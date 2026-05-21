(function () {
    "use strict";

    function copyText(text) {
        if (navigator.clipboard && navigator.clipboard.writeText) {
            return navigator.clipboard.writeText(text);
        }

        const input = document.createElement("textarea");
        input.value = text;
        input.setAttribute("readonly", "");
        input.style.position = "fixed";
        input.style.opacity = "0";
        document.body.appendChild(input);
        input.select();
        document.execCommand("copy");
        document.body.removeChild(input);
        return Promise.resolve();
    }

    function addCopyButton(target, latex, isDisplay) {
        if (!target || !latex || target.querySelector(".copy-latex-btn")) {
            return;
        }

        target.classList.add("copy-latex-target");
        if (isDisplay) {
            target.classList.add("copy-latex-display");
        }

        const btn = document.createElement("button");
        btn.className = "copy-latex-btn";
        btn.type = "button";
        btn.title = "Copy LaTeX";
        btn.setAttribute("aria-label", "Copy LaTeX");
        btn.textContent = "Copy";

        btn.addEventListener("click", function (event) {
            event.preventDefault();
            event.stopPropagation();

            copyText(latex).then(function () {
                btn.textContent = "OK";
                setTimeout(function () {
                    btn.textContent = "Copy";
                }, 1000);
            });
        });

        target.appendChild(btn);
    }

    function attachMathJax2Buttons() {
        if (!window.MathJax || !window.MathJax.Hub || !window.MathJax.Hub.getAllJax) {
            return false;
        }

        window.MathJax.Hub.getAllJax().forEach(function (jax) {
            const source = jax.originalText || "";
            const sourceScript = document.getElementById(jax.inputID);
            const isDisplay = !!(sourceScript && sourceScript.type.indexOf("mode=display") !== -1);
            const target = document.getElementById(jax.inputID + "-Frame");

            addCopyButton(target, source.trim(), isDisplay);
        });

        return true;
    }

    function attachScriptFallbackButtons() {
        document.querySelectorAll('script[type^="math/tex"]').forEach(function (script) {
            const latex = script.textContent.trim();
            const isDisplay = script.type.indexOf("mode=display") !== -1;
            const rendered = script.nextElementSibling;

            if (rendered && (rendered.classList.contains("MathJax") || rendered.tagName.toLowerCase() === "mjx-container")) {
                addCopyButton(rendered, latex, isDisplay);
            }
        });
    }

    function attachButtons() {
        const handledByMathJax = attachMathJax2Buttons();
        if (!handledByMathJax) {
            attachScriptFallbackButtons();
        }
    }

    function waitForMathJax(retries) {
        if (window.MathJax && window.MathJax.Hub && window.MathJax.Hub.Queue) {
            window.MathJax.Hub.Queue(attachButtons);
            return;
        }

        if (retries > 0) {
            window.setTimeout(function () {
                waitForMathJax(retries - 1);
            }, 500);
            return;
        }

        attachButtons();
    }

    function onReady() {
        waitForMathJax(20);
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", onReady);
    } else {
        onReady();
    }
}());
