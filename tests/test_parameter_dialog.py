import unittest
from unittest.mock import MagicMock, patch

from forensics_app.utils.parameter_dialog import ParameterField, ask_parameters


class ParameterDialogTests(unittest.TestCase):
    def test_no_threshold_input_accepts_values_above_field_maximum(self) -> None:
        parent = MagicMock()
        dialog = MagicMock()
        spinbox = MagicMock()
        spinbox.get.return_value = "18"
        apply_commands = []
        parent.wait_window.side_effect = lambda _: apply_commands[0]()

        def create_button(*args, **kwargs):
            if kwargs["text"] == "Apply":
                apply_commands.append(kwargs["command"])
            return MagicMock()

        with (
            patch(
                "forensics_app.utils.parameter_dialog.tk.Toplevel",
                return_value=dialog,
            ),
            patch("forensics_app.utils.parameter_dialog.tk.DoubleVar"),
            patch("forensics_app.utils.parameter_dialog.ttk.Label"),
            patch(
                "forensics_app.utils.parameter_dialog.ttk.Spinbox",
                return_value=spinbox,
            ) as spinbox_factory,
            patch("forensics_app.utils.parameter_dialog.ttk.Frame"),
            patch(
                "forensics_app.utils.parameter_dialog.ttk.Button",
                side_effect=create_button,
            ),
            patch("forensics_app.utils.parameter_dialog.messagebox.showerror") as error,
        ):
            result = ask_parameters(
                parent,
                "Parameters",
                (
                    ParameterField(
                        "sigma",
                        "Sigma",
                        0.0,
                        10.0,
                        0.1,
                        1.0,
                        show_slider=False,
                        no_threshold=True,
                    ),
                ),
            )

        spinbox_options = spinbox_factory.call_args.kwargs
        self.assertNotIn("from_", spinbox_options)
        self.assertNotIn("to", spinbox_options)
        self.assertEqual(result, {"sigma": 18.0})
        error.assert_not_called()
        dialog.destroy.assert_called_once()

    def test_no_threshold_input_still_enforces_minimum(self) -> None:
        parent = MagicMock()
        dialog = MagicMock()
        spinbox = MagicMock()
        spinbox.get.return_value = "-1"
        apply_commands = []
        parent.wait_window.side_effect = lambda _: apply_commands[0]()

        def create_button(*args, **kwargs):
            if kwargs["text"] == "Apply":
                apply_commands.append(kwargs["command"])
            return MagicMock()

        with (
            patch(
                "forensics_app.utils.parameter_dialog.tk.Toplevel",
                return_value=dialog,
            ),
            patch("forensics_app.utils.parameter_dialog.tk.DoubleVar"),
            patch("forensics_app.utils.parameter_dialog.ttk.Label"),
            patch(
                "forensics_app.utils.parameter_dialog.ttk.Spinbox",
                return_value=spinbox,
            ),
            patch("forensics_app.utils.parameter_dialog.ttk.Frame"),
            patch(
                "forensics_app.utils.parameter_dialog.ttk.Button",
                side_effect=create_button,
            ),
            patch("forensics_app.utils.parameter_dialog.messagebox.showerror") as error,
        ):
            result = ask_parameters(
                parent,
                "Parameters",
                (
                    ParameterField(
                        "sigma",
                        "Sigma",
                        0.0,
                        10.0,
                        0.1,
                        1.0,
                        show_slider=False,
                        no_threshold=True,
                    ),
                ),
            )

        self.assertIsNone(result)
        error.assert_called_once_with(
            "Parameters", "Sigma must be at least 0.", parent=dialog
        )

    def test_controls_can_be_selected_per_parameter(self) -> None:
        parent = MagicMock()
        with (
            patch("forensics_app.utils.parameter_dialog.tk.Toplevel") as toplevel,
            patch("forensics_app.utils.parameter_dialog.tk.DoubleVar"),
            patch("forensics_app.utils.parameter_dialog.tk.Scale") as scale,
            patch("forensics_app.utils.parameter_dialog.ttk.Label"),
            patch("forensics_app.utils.parameter_dialog.ttk.Spinbox") as spinbox,
            patch("forensics_app.utils.parameter_dialog.ttk.Frame"),
            patch("forensics_app.utils.parameter_dialog.ttk.Button"),
        ):
            ask_parameters(
                parent,
                "Parameters",
                (
                    ParameterField(
                        "input_only",
                        "Input only",
                        0.0,
                        1.0,
                        0.1,
                        0.5,
                        show_slider=False,
                    ),
                    ParameterField(
                        "slider_only",
                        "Slider only",
                        0.0,
                        1.0,
                        0.1,
                        0.5,
                        show_input=False,
                    ),
                    ("both", "Both", 0.0, 1.0, 0.1, 0.5),
                ),
            )

        self.assertEqual(scale.call_count, 2)
        self.assertEqual(spinbox.call_count, 2)
        self.assertEqual(
            spinbox.return_value.grid.call_args_list[0].kwargs["columnspan"],
            2,
        )
        toplevel.assert_called_once_with(parent)
        parent.wait_window.assert_called_once()

    def test_parameter_requires_at_least_one_control(self) -> None:
        with patch("forensics_app.utils.parameter_dialog.tk.Toplevel") as toplevel:
            with self.assertRaisesRegex(ValueError, "must have a control"):
                ask_parameters(
                    MagicMock(),
                    "Parameters",
                    (
                        ParameterField(
                            "empty",
                            "Empty",
                            0.0,
                            1.0,
                            0.1,
                            0.5,
                            show_slider=False,
                            show_input=False,
                        ),
                    ),
                )

        toplevel.assert_not_called()


if __name__ == "__main__":
    unittest.main()
