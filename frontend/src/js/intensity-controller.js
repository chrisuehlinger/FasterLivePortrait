/**
 * Intensity Controller
 * 
 * This script handles animation intensity updates from the director
 * To use in viewer pages, include this script and make sure to call:
 * setupIntensityControl(socket);
 * 
 * Your animation code can then use the global animationIntensity variable
 * to adjust animation parameters.
 */

// Global animation intensity value (0.0 to 1.0)
let animationIntensity = 1.0;

/**
 * Set up the intensity control by connecting it to a WebSocket
 * @param {WebSocket} socket - WebSocket connection to the server
 */
function setupIntensityControl(socket) {
    // Handle incoming intensity updates
    const originalOnMessage = socket.onmessage;
    
    socket.onmessage = function(event) {
        try {
            // First allow any existing handler to process the message
            if (originalOnMessage) {
                originalOnMessage.call(socket, event);
            }
            
            // Try to parse as JSON
            if (typeof event.data === 'string') {
                const data = JSON.parse(event.data);
                
                // Handle intensity updates
                if (data.action === 'intensity_update') {
                    animationIntensity = parseFloat(data.value);
                    console.log(`Animation intensity updated: ${animationIntensity}`);
                    
                    // Trigger a custom event that page code can listen for
                    const intensityEvent = new CustomEvent('intensityupdate', { 
                        detail: { intensity: animationIntensity } 
                    });
                    document.dispatchEvent(intensityEvent);
                }
            }
        } catch (e) {
            // Non-JSON messages are likely binary frames
            console.error('Error processing WebSocket message:', e);
        }
    };
    
    console.log('Intensity controller initialized');
}

// Example usage in animation code:
// 
// document.addEventListener('intensityupdate', function(e) {
//     const intensity = e.detail.intensity;
//     // Update animation parameters based on intensity
//     myAnimation.speed = baseSpeed * intensity;
//     myAnimation.amplitude = baseAmplitude * intensity;
// });
