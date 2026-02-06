I've created the matplotlib skill with all three files. Here's a summary of what was generated:

**SKILL.md** (main skill file - 98 lines)
- Quick start with Agg backend setup
- Basic figure/subplot patterns
- Key concepts table
- Common patterns: grouped bars, heatmaps, forest plots
- Links to reference files and related skills

**references/patterns.md** (137 lines)
- Backend and import configuration
- Color schemes: grayscale palette, seaborn diverging, semantic flag colors
- Chart types: grouped bars, horizontal bars, heatmaps, forest plots, KDE posteriors
- Axis and layout configuration
- Text and annotations
- Anti-patterns with WARNING sections: backend after import, inconsistent subplot indexing, missing plt.close()

**references/workflows.md** (148 lines)
- Publication export workflow (dual PDF/PNG)
- HTML report embedding with base64
- Multi-model comparison figures with dynamic grids
- Error-resilient plotting with fallbacks
- Figure metadata management
- Integration patterns with seaborn

The skill is tailored to this codebase's specific usage patterns:
- Agg backend for headless rendering
- 300 DPI dual-format export for publications
- Grayscale palettes for accessibility
- Base64 embedding for self-contained HTML reports
- Error fallbacks (violin→boxplot, KDE→histogram)
- Cross-references to related skills (seaborn, numpy, pandas, arviz)