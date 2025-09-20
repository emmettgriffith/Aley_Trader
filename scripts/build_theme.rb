#!/usr/bin/env ruby
# frozen_string_literal: true

require 'json'
require 'time'
require 'fileutils'

# Simple helper to blend a color with another
class Color
  def initialize(hex)
    hex = hex.delete('#')
    @r, @g, @b = hex.scan(/../).map { |component| component.to_i(16) }
  end

  def blend(other_hex, ratio)
    other = Color.new(other_hex)
    r = (@r * (1 - ratio) + other.r * ratio).round
    g = (@g * (1 - ratio) + other.g * ratio).round
    b = (@b * (1 - ratio) + other.b * ratio).round
    format('#%02X%02X%02X', r, g, b)
  end

  attr_reader :r, :g, :b
end

ocean_primary = '#050E1B'
deep_blue = '#0F1F33'
ite_blue = '#1F3654'
slate = '#2F4C6D'
silver = '#C7D3E3'
coral = '#FF6B6B'
seafoam = '#2ECC71'
sunrise = '#F7B731'
turquoise = '#4CD7D0'
ocean_glow = Color.new(ocean_primary).blend('#1C5D99', 0.55)

palette = {
  'primary_bg' => ocean_primary,
  'secondary_bg' => deep_blue,
  'accent_bg' => ocean_glow,
  'surface_bg' => slate,
  'card_bg' => Color.new(ocean_primary).blend('#112F4C', 0.45),
  'panel_overlay' => Color.new(deep_blue).blend('#061320', 0.35),
  'gridline' => Color.new('#FFFFFF').blend('#0A1933', 0.85),
  'divider' => Color.new('#FFFFFF').blend('#0A1933', 0.75),
  'text_primary' => '#F5FAFF',
  'text_secondary' => silver,
  'text_muted' => Color.new(silver).blend('#1F3654', 0.6),
  'text_accent' => '#76C7FF',
  'success' => seafoam,
  'danger' => coral,
  'warning' => sunrise,
  'info' => '#5DADE2',
  'highlight' => turquoise,
  'shadow' => '#030911'
}

gradients = {
  'primary' => [ocean_primary, deep_blue, ocean_glow],
  'button' => [Color.new('#4C74C9').blend('#1B3C73', 0.4), '#1B3C73'],
  'chart_background' => [Color.new('#01060F').blend(deep_blue, 0.6), deep_blue]
}

fonts = {
  'base_family' => '"Inter", "Segoe UI", "Helvetica Neue", sans-serif',
  'mono_family' => '"JetBrains Mono", "Fira Code", monospace',
  'title_weight' => 600,
  'body_weight' => 400,
  'small_caps' => 500
}

borders = {
  'radius_sm' => 6,
  'radius_lg' => 18,
  'width' => 1,
  'glow_width' => 2
}

lighting = {
  'card_shadow' => {
    'offset' => [0, 18],
    'blur' => 45,
    'spread' => -24,
    'color' => '#031627C8'
  },
  'hover_shadow' => {
    'offset' => [0, 12],
    'blur' => 32,
    'spread' => -16,
    'color' => '#0E4375AA'
  }
}

ui_states = {
  'hover' => Color.new('#76C7FF').blend('#5082FF', 0.4),
  'active' => Color.new('#76C7FF').blend('#1B98FF', 0.7),
  'focus_ring' => '#29E3FF',
  'selection_bg' => Color.new('#76C7FF').blend('#12243F', 0.35),
  'selection_fg' => '#FFFFFF'
}

signals = {
  'bullish' => '#24E1A6',
  'bearish' => '#FF5C8A',
  'neutral' => '#BDC3C7',
  'volume_up' => '#2ED8A0',
  'volume_down' => '#FF7A8A'
}

schema = {
  'generated_at' => Time.now.utc.iso8601,
  'description' => 'TradingView-inspired deep ocean palette with brand accents',
  'palette' => palette,
  'gradients' => gradients,
  'fonts' => fonts,
  'borders' => borders,
  'lighting' => lighting,
  'ui_states' => ui_states,
  'signals' => signals
}

output_path = ARGV[0] || File.join(__dir__, '..', 'ui', 'themes', 'tradingview_theme.json')
FileUtils.mkdir_p(File.dirname(output_path))
File.write(output_path, JSON.pretty_generate(schema))

puts "Theme written to #{output_path}"
