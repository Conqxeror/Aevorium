import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import { Colors, FontSizes, Spacing, BorderRadius } from '../theme';

const getEventConfig = (eventType) => {
  const type = eventType?.toLowerCase() || '';
  
  if (type.includes('privacy')) {
    return { icon: 'shield-lock', color: Colors.chartPurple };
  }
  if (type.includes('model') || type.includes('saved')) {
    return { icon: 'content-save', color: Colors.chartBlue };
  }
  if (type.includes('generation') || type.includes('generate')) {
    return { icon: 'chart-bar', color: Colors.chartGreen };
  }
  if (type.includes('training') || type.includes('train')) {
    return { icon: 'cog', color: Colors.chartOrange };
  }
  if (type.includes('error') || type.includes('fail')) {
    return { icon: 'alert-circle', color: Colors.danger };
  }
  return { icon: 'circle-small', color: Colors.textSecondary };
};

export default function ActivityItem({ timestamp, eventType, details }) {
  const config = getEventConfig(eventType);
  const formattedTime = timestamp?.slice(0, 19) || 'Unknown time';

  return (
    <View style={styles.container}>
      <View style={[styles.iconContainer, { backgroundColor: `${config.color}20` }]}>
        <MaterialCommunityIcons name={config.icon} size={16} color={config.color} />
      </View>
      <View style={styles.content}>
        <View style={styles.header}>
          <Text style={styles.eventType}>{eventType || 'Unknown'}</Text>
          <Text style={styles.timestamp}>{formattedTime}</Text>
        </View>
        {details && Object.keys(details).length > 0 && (
          <Text style={styles.details} numberOfLines={2}>
            {Object.entries(details).map(([k, v]) => `${k}: ${v}`).join(' • ')}
          </Text>
        )}
      </View>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    paddingVertical: Spacing.sm,
    borderBottomWidth: 1,
    borderBottomColor: Colors.border,
  },
  iconContainer: {
    width: 32,
    height: 32,
    borderRadius: BorderRadius.round,
    justifyContent: 'center',
    alignItems: 'center',
    marginRight: Spacing.sm,
  },
  content: {
    flex: 1,
  },
  header: {
    flexDirection: 'row',
    justifyContent: 'space-between',
    alignItems: 'center',
    marginBottom: 2,
  },
  eventType: {
    fontSize: FontSizes.md,
    fontWeight: '600',
    color: Colors.text,
    flex: 1,
  },
  timestamp: {
    fontSize: FontSizes.xs,
    color: Colors.textMuted,
  },
  details: {
    fontSize: FontSizes.sm,
    color: Colors.textSecondary,
    marginTop: 2,
  },
});
