from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock

from forensics_app.ui.main_window import MainWindow


class MainWindowTests(unittest.TestCase):
    def setUp(self) -> None:
        self.window = MainWindow.__new__(MainWindow)
        self.window.sidebar_canvas = MagicMock()
        self.window.sidebar_canvas.winfo_rootx.return_value = 10
        self.window.sidebar_canvas.winfo_rooty.return_value = 20
        self.window.sidebar_canvas.winfo_width.return_value = 200
        self.window.sidebar_canvas.winfo_height.return_value = 300

    def test_mousewheel_scrolls_sidebar_when_pointer_is_over_canvas(self) -> None:
        event = SimpleNamespace(x_root=100, y_root=100, num=None, delta=120)

        result = self.window._on_sidebar_mousewheel(event)

        self.assertEqual(result, "break")
        self.window.sidebar_canvas.yview_scroll.assert_called_once_with(-1, "units")

    def test_linux_mousewheel_scrolls_down_when_pointer_is_over_canvas(self) -> None:
        event = SimpleNamespace(x_root=100, y_root=100, num=5, delta=0)

        result = self.window._on_sidebar_mousewheel(event)

        self.assertEqual(result, "break")
        self.window.sidebar_canvas.yview_scroll.assert_called_once_with(1, "units")

    def test_mousewheel_does_not_scroll_sidebar_when_pointer_is_outside(self) -> None:
        event = SimpleNamespace(x_root=500, y_root=100, num=None, delta=120)

        result = self.window._on_sidebar_mousewheel(event)

        self.assertIsNone(result)
        self.window.sidebar_canvas.yview_scroll.assert_not_called()


if __name__ == "__main__":
    unittest.main()
