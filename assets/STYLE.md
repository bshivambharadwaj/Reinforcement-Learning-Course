# Course visual style

The course uses **ink-plum, antique gold, sage, and warm paper**. The light course banner, diagrams, and equation cards share this restrained palette, with labels supplying meaning independently of color.

| Use | Color |
| --- | --- |
| Headings and equation ink | `#292332` |
| Paper background | `#F5F1E8` |
| Decisions and numbered arrow markers | `#AF824A` |
| Environment and outcomes | `#768165` |
| Learning accent | `#87708C` |
| Secondary text | `#766C7C` |

## Diagrams

- Every canvas is 760 pixels wide and uses a scalable SVG viewBox.
- README embeds specify width only; GitHub can scale images down while preserving aspect ratio.
- Compact folded layouts replace long vertical chains. Card labels are wrapped using actual font metrics and checked against card boundaries.
- Connectors route around node interiors. Numbered arrows refer to notes beneath the chart.
- Fonts are exported as vector paths, so the appearance does not depend on locally installed fonts.

Regenerate with Python and matplotlib:

```bash
python3 assets/diagrams/generate_diagrams.py
```

The generator updates SVGs and writes high-resolution PNG inspection copies under `/tmp/rl-diagram-previews/`.

## Equations

Display equations are pre-rendered SVGs. Inline notation uses ordinary Unicode and HTML `sup` / `sub` elements. The README does not depend on a Markdown math parser, MathJax macros, or Mermaid.

The editable LaTeX is in `equations/equations.json`. Each entry identifies the course section. To update an equation, edit its LaTeX, regenerate the SVG, and update its README image alt text to match.

Regenerate with Python, LaTeX (amsmath, amssymb, preview), and dvisvgm:

```bash
python3 assets/equations/generate_equations.py
```

Equation images use outlined fonts, a warm-paper background, and dark ink for legibility on either GitHub page theme. The original LaTeX is also included in each SVG's description.
