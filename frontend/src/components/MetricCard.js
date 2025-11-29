import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import { Colors, FontSizes, Spacing, BorderRadius, Shadows } from '../theme';

export default function MetricCard({ 
  title, 
  value, 
  subtitle, 
  icon, 
  iconColor = Colors.primary,
  trend,
  trendDirection = 'up',
  accentColor = Colors.primary 
}) {
  const getTrendColor = () => {
    if (trendDirection === 'up') return Colors.success;
    if (trendDirection === 'down') return Colors.danger;
    return Colors.textSecondary;
  };

  return (
    <View style={styles.container}>
      <View style={styles.header}>
        <View style={[styles.iconContainer, { backgroundColor: `${iconColor}15` }]}>
          <MaterialCommunityIcons name={icon} size={22} color={iconColor} />
        </View>
        {trend && (
          <View style={[styles.trendBadge, { backgroundColor: `${getTrendColor()}15` }]}>
            <MaterialCommunityIcons 
              name={trendDirection === 'up' ? 'arrow-up' : trendDirection === 'down' ? 'arrow-down' : 'minus'}
              size={12} 
              color={getTrendColor()} 
            />
            <Text style={[styles.trendText, { color: getTrendColor() }]}>{trend}</Text>
          </View>
        )}
      </View>
      
      <View style={styles.content}>
        <Text style={styles.value} numberOfLines={1} adjustsFontSizeToFit>{value}</Text>
        <Text style={styles.title} numberOfLines={1}>{title}</Text>
        {subtitle && <Text style={styles.subtitle} numberOfLines={1}>{subtitle}</Text>}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    backgroundColor: Colors.surface,
    borderRadius: BorderRadius.xl,
    padding: Spacing.md,
    ...Shadows.sm,
    flex: 1,
    margin: Spacing.xs,
    borderWidth: 1,
    borderColor: Colors.border,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'flex-start',
    marginBottom: Spacing.md,
  },
  iconContainer: {
    borderRadius: BorderRadius.lg,
    padding: Spacing.sm,
    justifyContent: 'center',
    alignItems: 'center',
  },
  trendBadge: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingHorizontal: 6,
    paddingVertical: 2,
    borderRadius: BorderRadius.round,
  },
  trendText: {
    fontSize: FontSizes.xs,
    fontWeight: '700',
    marginLeft: 2,
  },
  content: {
    justifyContent: 'flex-end',
  },
  value: {
    fontSize: FontSizes.xl,
    fontWeight: 'bold',
    color: Colors.text,
    marginBottom: 2,
  },
  title: {
    fontSize: FontSizes.sm,
    color: Colors.textSecondary,
    fontWeight: '500',
    marginBottom: 2,
  },
  subtitle: {
    fontSize: FontSizes.xs,
    color: Colors.textMuted,
  },
});
