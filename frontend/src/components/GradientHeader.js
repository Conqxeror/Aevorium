import React from 'react';
import { View, Text, StyleSheet, Platform } from 'react-native';
import { LinearGradient } from 'expo-linear-gradient';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import { Colors, FontSizes, Spacing, BorderRadius, Shadows } from '../theme';

export default function GradientHeader({ title, subtitle, icon }) {
  return (
    <View style={styles.wrapper}>
      <LinearGradient
        colors={[Colors.primary, Colors.primaryDark]}
        start={{ x: 0, y: 0 }}
        end={{ x: 1, y: 1 }}
        style={styles.container}
      >
        <View style={styles.patternOverlay} />
        <View style={styles.content}>
          {icon && (
            <View style={styles.iconContainer}>
              <MaterialCommunityIcons 
                name={icon} 
                size={32} 
                color={Colors.text} 
              />
            </View>
          )}
          <View style={styles.textContainer}>
            <Text style={styles.title}>{title}</Text>
            {subtitle && <Text style={styles.subtitle}>{subtitle}</Text>}
          </View>
        </View>
      </LinearGradient>
    </View>
  );
}

const styles = StyleSheet.create({
  wrapper: {
    marginBottom: Spacing.md,
    ...Shadows.lg,
    backgroundColor: Colors.background, // Prevent shadow artifacts
  },
  container: {
    paddingTop: Platform.OS === 'ios' ? 60 : 50, // Account for status bar
    paddingBottom: Spacing.xl,
    paddingHorizontal: Spacing.lg,
    borderBottomLeftRadius: BorderRadius.xl,
    borderBottomRightRadius: BorderRadius.xl,
    position: 'relative',
    overflow: 'hidden',
  },
  patternOverlay: {
    ...StyleSheet.absoluteFillObject,
    backgroundColor: 'rgba(255, 255, 255, 0.03)',
    transform: [{ rotate: '-10deg' }, { scale: 1.5 }],
    // Simple pattern effect could be added here if we had an image, 
    // but for now just a subtle overlay
  },
  content: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  iconContainer: {
    marginRight: Spacing.md,
    backgroundColor: 'rgba(255, 255, 255, 0.15)',
    padding: Spacing.sm,
    borderRadius: BorderRadius.lg,
  },
  textContainer: {
    flex: 1,
  },
  title: {
    fontSize: FontSizes.xxl,
    fontWeight: 'bold',
    color: Colors.text,
    letterSpacing: 0.5,
  },
  subtitle: {
    fontSize: FontSizes.sm,
    color: 'rgba(255, 255, 255, 0.8)',
    marginTop: 4,
    lineHeight: 20,
  },
});
