// Custom JavaScript for Brain Tumor Detection App

// Auto-hide alerts after 5 seconds
document.addEventListener('DOMContentLoaded', function() {
    const alerts = document.querySelectorAll('.alert');
    alerts.forEach(function(alert) {
        setTimeout(function() {
            const bsAlert = new bootstrap.Alert(alert);
            bsAlert.close();
        }, 5000);
    });
});

// File size validation
function validateFileSize(input) {
    const maxSize = 16 * 1024 * 1024; // 16MB
    if (input.files[0] && input.files[0].size > maxSize) {
        alert('File size exceeds 16MB limit!');
        input.value = '';
        return false;
    }
    return true;
}
