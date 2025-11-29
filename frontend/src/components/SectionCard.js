import React, { useState } from 'react';
import { View, Text, StyleSheet, TouchableOpacity, Animated } from 'react-native';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import { Colors, FontSizes, Spacing, BorderRadius, Shadows } from '../theme';

export default function SectionCard({ 
  title, 
  icon, 
  children, 
  collapsible = false, 
  defaultExpanded = true,
  headerRight,
  noPadding = false,
}) {
  const [expanded, setExpanded] = useState(defaultExpanded);

  return (
    <View style={styles.container}>
      <TouchableOpacity 
        style={styles.header}
        onPress={() => collapsible && setExpanded(!expanded)}
        disabled={!collapsible}
        activeOpacity={collapsible ? 0.7 : 1}
      >
        <View style={styles.headerLeft}>
          {icon && (
            <View style={styles.iconContainer}>
              <MaterialCommunityIcons 
                name={icon} 
                size={18} 
                color={Colors.primary} 
              />
            </View>
          )}
          <Text style={styles.title}>{title}</Text>
        </View>
        <View style={styles.headerRight}>
          {headerRight}
          {collapsible && (
            <MaterialCommunityIcons 
              name={expanded ? 'chevron-up' : 'chevron-down'} 
              size={20} 
              color={Colors.textSecondary} 
            />
          )}
        </View>
      </TouchableOpacity>
      {(!collapsible || expanded) && (
        <View style={[styles.content, noPadding && styles.noPadding]}>
          {children}
        </View>
      )}
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    backgroundColor: Colors.surface,
    borderRadius: BorderRadius.xl,
    marginVertical: Spacing.sm,
    borderWidth: 1,
    borderColor: Colors.border,
    overflow: 'hidden',
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    padding: Spacing.md,
    backgroundColor: Colors.surfaceLight,
    borderBottomWidth: 1,
    borderBottomColor: Colors.border,
  },
  headerLeft: {
    flexDirection: 'row',
    alignItems: 'center',
    flex: 1,
  },
  headerRight: {
    flexDirection: 'row',
    alignItems: 'center',
  },
  iconContainer: {
    marginRight: Spacing.sm,
    backgroundColor: `${Colors.primary}15`,
    padding: 6,
    borderRadius: BorderRadius.md,
  },
  title: {
    fontSize: FontSizes.md,
    fontWeight: '600',
    color: Colors.text,
    letterSpacing: 0.3,
  },
  content: {
    padding: Spacing.md,
  },
  noPadding: {
    padding: 0,
  },
});
