// script.js

document.addEventListener("DOMContentLoaded", function() {
    const submitButton = document.querySelector('.submit-button');

    submitButton.addEventListener('click', function() {
        submitButton.textContent = 'Predicting...';
        submitButton.disabled = true;
    });
});
