import pyray as rl
from collections.abc import Callable
from openpilot.system.ui.lib.application import gui_app, FontWeight
from openpilot.system.ui.lib.multilang import tr
from openpilot.system.ui.widgets import Widget, DialogResult
from openpilot.system.ui.widgets.button import Button, ButtonStyle
from openpilot.system.ui.widgets.label import gui_label
from openpilot.system.ui.widgets.scroller_tici import Scroller

# Constants
MARGIN = 50
TITLE_FONT_SIZE = 70
ITEM_HEIGHT = 135
BUTTON_SPACING = 50
BUTTON_HEIGHT = 160
ITEM_SPACING = 50
LIST_ITEM_SPACING = 25


class MultiOptionDialog(Widget):
  def __init__(self, title, options, current="", option_font_weight=FontWeight.MEDIUM, callback: Callable[[DialogResult], None] | None = None):
    super().__init__()
    self.title = title
    self.options = options
    self.current = current
    self.selection = current
    self._callback = callback

    # Create scroller with option buttons
    self.option_buttons = [Button(option, click_callback=lambda opt=option: self._on_option_clicked(opt),
                                  font_weight=option_font_weight,
                                  text_alignment=rl.GuiTextAlignment.TEXT_ALIGN_LEFT, button_style=ButtonStyle.NORMAL,
                                  text_padding=50, elide_right=True) for option in options]
    self.scroller = Scroller(self.option_buttons, spacing=LIST_ITEM_SPACING)

    self.cancel_button = Button(lambda: tr("Cancel"), click_callback=lambda: self._set_result(DialogResult.CANCEL))
    self.select_button = Button(lambda: tr("Select"), click_callback=lambda: self._set_result(DialogResult.CONFIRM), button_style=ButtonStyle.PRIMARY)

  def _set_result(self, result: DialogResult):
    gui_app.pop_widget()
    if self._callback:
      self._callback(result)

  def _on_option_clicked(self, option):
    self.selection = option

  def _render(self, rect):
    margin = min(MARGIN, max(12, min(rect.width, rect.height) * 0.08))
    dialog_rect = rl.Rectangle(rect.x + margin, rect.y + margin, max(1, rect.width - 2 * margin), max(1, rect.height - 2 * margin))
    rl.draw_rectangle_rounded(dialog_rect, 0.02, 20, rl.Color(30, 30, 30, 255))

    content_rect = rl.Rectangle(dialog_rect.x + margin, dialog_rect.y + margin,
                                max(1, dialog_rect.width - 2 * margin), max(1, dialog_rect.height - 2 * margin))

    gui_label(rl.Rectangle(content_rect.x, content_rect.y, content_rect.width, TITLE_FONT_SIZE), self.title, 70, font_weight=FontWeight.BOLD)

    # Options area
    options_y = content_rect.y + TITLE_FONT_SIZE + ITEM_SPACING
    button_height = min(BUTTON_HEIGHT, max(120, content_rect.height * 0.28))
    options_h = max(1, content_rect.height - TITLE_FONT_SIZE - button_height - 2 * ITEM_SPACING)
    options_rect = rl.Rectangle(content_rect.x, options_y, content_rect.width, options_h)

    # Update button styles and set width based on selection
    for i, option in enumerate(self.options):
      selected = option == self.selection
      button = self.option_buttons[i]
      button.set_button_style(ButtonStyle.PRIMARY if selected else ButtonStyle.NORMAL)
      button.set_rect(rl.Rectangle(0, 0, options_rect.width, ITEM_HEIGHT))

    self.scroller.render(options_rect)

    # Buttons
    button_y = content_rect.y + content_rect.height - button_height
    button_spacing = min(BUTTON_SPACING, max(12, content_rect.width * 0.05))
    button_w = max(1, (content_rect.width - button_spacing) / 2)

    cancel_rect = rl.Rectangle(content_rect.x, button_y, button_w, button_height)
    self.cancel_button.render(cancel_rect)

    select_rect = rl.Rectangle(content_rect.x + button_w + button_spacing, button_y, button_w, button_height)
    self.select_button.set_enabled(self.selection != self.current)
    self.select_button.render(select_rect)
