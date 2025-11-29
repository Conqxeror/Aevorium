/**
 * Aevorium Theme Configuration
 * Matches the Streamlit dashboard's visual style
 */

export const Colors = {
  // Primary gradient colors (matching Streamlit header)
  primary: '#1f77b4',
  primaryDark: '#1557a0',
  secondary: '#2ca02c',
  
  // Background colors
  background: '#0e1117',
  surface: '#1e2130',
  surfaceLight: '#262b3d',
  cardBackground: '#1a1f2e',
  
  // Text colors
  text: '#fafafa',
  textSecondary: '#a3a8b8',
  textMuted: '#6b7280',
  
  // Status colors
  success: '#28a745',
  successLight: '#d4edda',
  warning: '#ffc107',
  warningLight: '#fff3cd',
  danger: '#dc3545',
  dangerLight: '#f8d7da',
  info: '#17a2b8',
  infoLight: '#d1ecf1',
  
  // Chart colors
  chartBlue: '#1f77b4',
  chartGreen: '#2ca02c',
  chartOrange: '#ff7f0e',
  chartRed: '#d62728',
  chartPurple: '#9467bd',
  
  // Accent
  accent: '#6366f1',
  accentLight: '#818cf8',
  
  // Border
  border: '#2d3748',
  borderLight: '#4a5568',
};

export const Spacing = {
  xs: 4,
  sm: 8,
  md: 16,
  lg: 24,
  xl: 32,
  xxl: 48,
};

export const FontSizes = {
  xs: 10,
  sm: 12,
  md: 14,
  lg: 16,
  xl: 20,
  xxl: 24,
  xxxl: 32,
  display: 40,
};

export const BorderRadius = {
  sm: 4,
  md: 8,
  lg: 12,
  xl: 16,
  round: 999,
};

export const Shadows = {
  sm: {
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 1 },
    shadowOpacity: 0.2,
    shadowRadius: 2,
    elevation: 2,
  },
  md: {
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 2 },
    shadowOpacity: 0.25,
    shadowRadius: 4,
    elevation: 4,
  },
  lg: {
    shadowColor: '#000',
    shadowOffset: { width: 0, height: 4 },
    shadowOpacity: 0.3,
    shadowRadius: 8,
    elevation: 8,
  },
};

// React Native Paper theme configuration
export const PaperTheme = {
  dark: true,
  mode: 'adaptive',
  roundness: BorderRadius.md,
  colors: {
    primary: Colors.primary,
    primaryContainer: Colors.primaryDark,
    secondary: Colors.secondary,
    secondaryContainer: Colors.secondary,
    tertiary: Colors.accent,
    tertiaryContainer: Colors.accentLight,
    surface: Colors.surface,
    surfaceVariant: Colors.surfaceLight,
    surfaceDisabled: Colors.surface,
    background: Colors.background,
    error: Colors.danger,
    errorContainer: Colors.dangerLight,
    onPrimary: Colors.text,
    onPrimaryContainer: Colors.text,
    onSecondary: Colors.text,
    onSecondaryContainer: Colors.text,
    onTertiary: Colors.text,
    onTertiaryContainer: Colors.text,
    onSurface: Colors.text,
    onSurfaceVariant: Colors.textSecondary,
    onSurfaceDisabled: Colors.textMuted,
    onError: Colors.text,
    onErrorContainer: Colors.danger,
    onBackground: Colors.text,
    outline: Colors.border,
    outlineVariant: Colors.borderLight,
    inverseSurface: Colors.text,
    inverseOnSurface: Colors.background,
    inversePrimary: Colors.primaryDark,
    shadow: '#000000',
    scrim: '#000000',
    backdrop: 'rgba(0, 0, 0, 0.5)',
    elevation: {
      level0: 'transparent',
      level1: Colors.surface,
      level2: Colors.surfaceLight,
      level3: Colors.cardBackground,
      level4: Colors.cardBackground,
      level5: Colors.cardBackground,
    },
  },
};

export default {
  Colors,
  Spacing,
  FontSizes,
  BorderRadius,
  Shadows,
  PaperTheme,
};
