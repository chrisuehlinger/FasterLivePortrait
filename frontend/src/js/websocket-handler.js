/**
 * Unified WebSocket Message Handler
 * 
 * This module provides a consistent way to handle WebSocket messages across
 * actor, director, and viewer pages. It standardizes message handling and
 * dispatches appropriate events that pages can listen for.
 */

// Default animation intensity value (0.0 to 1.0)
let animationIntensity = 1.0;

/**
 * Message action types that can be received from the server
 */
const MESSAGE_TYPES = {
  SOURCE_SWITCHED: 'source_switched',
  INTENSITY_UPDATE: 'intensity_update',
  CONNECTED: 'connected',
  HEARTBEAT_ACK: 'heartbeat_ack',
  FRAME_PROCESSED: 'frame_processed',
  PAUSED: 'paused',
  RESUMED: 'resumed',
  ERROR: 'error',
  SET_MOTAL: 'set_motal',
  // Add any additional message types here
  STATUS_UPDATE: 'status'
};

/**
 * Setup unified WebSocket message handling
 * @param {WebSocket} socket - WebSocket connection to wrap
 * @param {Object} options - Configuration options
 * @param {Function} options.onBinaryData - Handler for binary data (frames)
 * @param {Function} options.onSourceSwitched - Handler for source switching
 * @param {Function} options.onIntensityUpdate - Handler for intensity updates
 * @param {Function} options.onConnectionStatus - Handler for connection status updates
 * @param {Function} options.onError - Handler for error messages
 * @returns {WebSocket} The enhanced WebSocket object
 */
function setupWebSocketHandler(socket, options = {}) {
  // Store the original handlers
  const originalOnMessage = socket.onmessage;
  const originalOnOpen = socket.onopen;
  const originalOnClose = socket.onclose;
  const originalOnError = socket.onerror;
  
  // Set up heartbeat interval
  let heartbeatInterval = null;
  
  // Enhanced onopen handler
  socket.onopen = function(event) {
    console.log('WebSocket connection established');
    
    // Call the original handler if it exists
    if (originalOnOpen) {
      originalOnOpen.call(socket, event);
    }
    
    // Set up heartbeat every 30 seconds to keep connection alive
    heartbeatInterval = setInterval(() => {
      if (socket && socket.readyState === WebSocket.OPEN) {
        socket.send('heartbeat');
      }
    }, 30000);
    
    // Dispatch an event for connection
    document.dispatchEvent(new CustomEvent('websocket:open', { detail: { event } }));
  };
  
  // Enhanced onclose handler
  socket.onclose = function(event) {
    console.log('WebSocket connection closed');
    
    // Clear heartbeat interval
    if (heartbeatInterval) {
      clearInterval(heartbeatInterval);
    }
    
    // Call the original handler if it exists
    if (originalOnClose) {
      originalOnClose.call(socket, event);
    }
    
    // Dispatch an event for disconnection
    document.dispatchEvent(new CustomEvent('websocket:close', { 
      detail: { 
        event,
        wasClean: event.wasClean,
        code: event.code,
        reason: event.reason
      } 
    }));
  };
  
  // Enhanced onerror handler
  socket.onerror = function(error) {
    console.error('WebSocket error:', error);
    
    // Call the original handler if it exists
    if (originalOnError) {
      originalOnError.call(socket, error);
    }
    
    // Call custom error handler if provided
    if (options.onError) {
      options.onError(error);
    }
    
    // Dispatch an event for error
    document.dispatchEvent(new CustomEvent('websocket:error', { detail: { error } }));
  };
  
  // Enhanced onmessage handler
  socket.onmessage = function(event) {
    // Handle binary data (frames)
    if (typeof event.data !== 'string') {
      // If there's a specific binary handler, use it
      // Handle binary data
      if (options.onBinaryData) {
        options.onBinaryData(event.data);
      }
          
      // Dispatch a binary data event
      document.dispatchEvent(new CustomEvent('websocket:binary', { 
        detail: { 
          data: event.data,
          timestamp: Date.now()
        } 
      }));
      
      // Also call the original handler for backward compatibility
      if (originalOnMessage) {
        originalOnMessage.call(socket, event);
      }
      
      return;
    }
    
    // For text messages, try to parse as JSON
    try {
      const data = JSON.parse(event.data);
      console.log('Received message:', data);
      
      // Process different message types - checking action field first, then status for backward compatibility
      const messageType = data.action || data.status;
      console.log('Message type:', messageType, 'Data:', data);
      
      // Ensure data has consistent field names for event handling
      const normalizedData = {
        ...data,
        // Common fields that might be in different formats
        action: data.action || data.status,
        message: data.message || "",
        sourceIndex: data.current_source,
        sourceName: data.source_name,
        intensity: data.action === "intensity_update" ? data.value : undefined,
        motalEnabled: data.action === "set_motal" ? (data.value === true || data.value === "true" || data.value === 1 || data.value === "1") : undefined
      };
      
      if (messageType) {
        switch (messageType) {
          case MESSAGE_TYPES.SOURCE_SWITCHED:
            console.log(`Source switched to ${data.current_source}: ${data.source_name}`);
            
            // Call specific handler if provided
            if (options.onSourceSwitched) {
              options.onSourceSwitched(data);
            }
            
            // Dispatch a source switched event
            document.dispatchEvent(new CustomEvent('websocket:source_switched', { 
              detail: { 
                sourceIndex: data.current_source,
                sourceName: data.source_name,
                initiatedBy: data.initiated_by || 'unknown',
                // Pass original data structure for backward compatibility
                current_source: data.current_source,
                source_name: data.source_name,
                // Pass the complete normalized data
                ...normalizedData
              } 
            }));
            break;
            
          case MESSAGE_TYPES.CONNECTED:
            console.log(`Connected to session: ${data.session_id}`);
            
            // Call status handler if provided
            if (options.onConnectionStatus) {
              options.onConnectionStatus(normalizedData);
            }
            
            // Dispatch a connection status event
            document.dispatchEvent(new CustomEvent('websocket:status', { 
              detail: normalizedData 
            }));
            
            // Also dispatch a more specific connected event
            document.dispatchEvent(new CustomEvent('websocket:connected', { 
              detail: normalizedData 
            }));
            break;
            
          case MESSAGE_TYPES.HEARTBEAT_ACK:
            // Heartbeat acknowledgment, typically no action needed
            console.debug('Heartbeat acknowledged');
            break;
            
          case MESSAGE_TYPES.ERROR:
            console.error(`Server error: ${data.message}`);
            
            // Call error handler if provided
            if (options.onError) {
              options.onError(data);
            }
            
            // Dispatch an error event
            document.dispatchEvent(new CustomEvent('websocket:error', { 
              detail: { data } 
            }));
            break;
            
          case MESSAGE_TYPES.PAUSED:
          case MESSAGE_TYPES.RESUMED:
            console.log(`Stream ${messageType}: ${data.message}`);
            
            // Dispatch a pause/resume event
            document.dispatchEvent(new CustomEvent(`websocket:${messageType}`, { 
              detail: { data } 
            }));
            break;
            
          case MESSAGE_TYPES.INTENSITY_UPDATE:
              // Handle intensity updates
              animationIntensity = parseFloat(data.value);
              console.log(`Animation intensity updated: ${animationIntensity}`);
            
              // Call specific handler if provided
              if (options.onIntensityUpdate) {
                options.onIntensityUpdate(animationIntensity);
              }
            
              // Dispatch an intensity update event
              document.dispatchEvent(new CustomEvent('websocket:intensity_update', { 
                detail: { 
                  intensity: animationIntensity,
                  initiatedBy: data.initiated_by || 'unknown',
                  // Pass original value for backward compatibility
                  value: animationIntensity,
                  // Pass the complete normalized data
                  ...normalizedData
                } 
              }));
              break;
              
          case MESSAGE_TYPES.SET_MOTAL:
              console.log(`Motal status updated: ${data.value}`);
              
              // Dispatch a motal update event
              document.dispatchEvent(new CustomEvent('websocket:set_motal', { 
                detail: { 
                  motalEnabled: data.value === true || data.value === "true" || data.value === 1 || data.value === "1",
                  // Pass the complete normalized data
                  ...normalizedData
                } 
              }));
              break;
            
            default:
              // For any other action type, dispatch two events:
              // 1. A specific event for the action type
              console.log(`Dispatching event websocket:${messageType} with data:`, normalizedData);
              document.dispatchEvent(new CustomEvent(`websocket:${messageType}`, { 
                detail: normalizedData
              }));
              // 2. A generic message event for all handlers
              document.dispatchEvent(new CustomEvent('websocket:message', { 
                detail: normalizedData
              }));
              break;
          }
        } else {
        // For any other message format, dispatch a generic message event
        console.log('Unhandled message format:', data);
        document.dispatchEvent(new CustomEvent('websocket:message', { 
          detail: data 
        }));
      }
      
    } catch (e) {
      console.warn('Received non-JSON message:', event.data);
      
      // For non-JSON messages, dispatch a text event
      document.dispatchEvent(new CustomEvent('websocket:text', { 
        detail: { text: event.data } 
      }));
    }
    
    // Also call the original handler for backward compatibility
    if (originalOnMessage) {
      originalOnMessage.call(socket, event);
    }
  };
  
  // Add a convenient send method for JSON data
  socket.sendJson = function(data) {
    if (socket.readyState === WebSocket.OPEN) {
      socket.send(JSON.stringify(data));
      return true;
    }
    return false;
  };
  
  return socket;
}

/**
 * Get the current animation intensity value
 * @returns {number} The current animation intensity (0.0 to 1.0)
 */
function getAnimationIntensity() {
  return animationIntensity;
}

/**
 * Send an animation intensity update to the server
 * @param {WebSocket} socket - WebSocket connection to the server
 * @param {number} intensity - New intensity value (0.0 to 1.0)
 */
function sendIntensityUpdate(socket, intensity) {
  if (socket && socket.readyState === WebSocket.OPEN) {
    const message = {
      action: 'intensity_update',
      value: parseFloat(intensity)
    };
    socket.send(JSON.stringify(message));
    return true;
  }
  return false;
}

// Export the module functions
export {
  setupWebSocketHandler,
  getAnimationIntensity,
  sendIntensityUpdate,
  MESSAGE_TYPES
};