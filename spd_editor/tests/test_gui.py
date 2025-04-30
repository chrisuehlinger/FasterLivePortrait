#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Tests for the graphical user interface of SPD Editor.

Since GUI testing requires user interaction or complex simulation,
these tests focus on unit testing the non-interactive components 
and basic initialization.
"""

import os
import sys
import unittest
from unittest import mock

# Mock all external dependencies first
sys.modules['tkinter'] = mock.MagicMock()
sys.modules['tkinter.ttk'] = mock.MagicMock() 
sys.modules['tkinter.filedialog'] = mock.MagicMock()
sys.modules['tkinter.messagebox'] = mock.MagicMock()
sys.modules['tkinter.font'] = mock.MagicMock()
sys.modules['tkinter.colorchooser'] = mock.MagicMock()
sys.modules['tkinter.simpledialog'] = mock.MagicMock()

sys.modules['cv2'] = mock.MagicMock()
sys.modules['numpy'] = mock.MagicMock()
sys.modules['PIL'] = mock.MagicMock()
sys.modules['PIL.Image'] = mock.MagicMock()
sys.modules['PIL.ImageTk'] = mock.MagicMock()

sys.modules['matplotlib'] = mock.MagicMock()
sys.modules['matplotlib.figure'] = mock.MagicMock()
sys.modules['matplotlib.pyplot'] = mock.MagicMock()
sys.modules['matplotlib.backends.backend_tkagg'] = mock.MagicMock()
sys.modules['mpl_toolkits'] = mock.MagicMock()
sys.modules['mpl_toolkits.mplot3d'] = mock.MagicMock()

# Mock the relevant components from spd_editor
sys.modules['spd_editor.utils'] = mock.MagicMock()
sys.modules['spd_editor.utils.visualization'] = mock.MagicMock()
sys.modules['spd_editor.spd.reader'] = mock.MagicMock()
sys.modules['spd_editor.spd.writer'] = mock.MagicMock()
sys.modules['spd_editor.spd.format'] = mock.MagicMock() 
sys.modules['spd_editor.spd.validator'] = mock.MagicMock()


class TestGUINoTkinter(unittest.TestCase):
    """Basic test cases for the GUI module."""
    
    def test_gui_module_imports(self):
        """Test that the GUI module can be imported without errors."""
        try:
            from spd_editor import gui
            self.assertTrue(hasattr(gui, 'main'))
        except ImportError as e:
            self.fail(f"Failed to import GUI module: {e}")
    
    def test_main_function(self):
        """Test the main function with mocked dependencies."""
        from spd_editor import gui
        
        # Mock the GUI components
        with mock.patch('spd_editor.gui.TKINTER_AVAILABLE', True), \
             mock.patch('spd_editor.gui.tk.Tk') as mock_tk, \
             mock.patch('spd_editor.gui.SPDEditorApp') as mock_app:
            
            # Mock the Tk instance
            mock_tk_instance = mock.MagicMock()
            mock_tk.return_value = mock_tk_instance
            
            # Call the main function
            gui.main()
            
            # Verify the application was created and mainloop was called
            mock_tk.assert_called_once()
            mock_app.assert_called_once()
            mock_tk_instance.mainloop.assert_called_once()


if __name__ == '__main__':
    unittest.main()