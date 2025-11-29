import React from 'react';
import { View, Text, StyleSheet } from 'react-native';
import { MaterialCommunityIcons } from '@expo/vector-icons';
import { Colors, FontSizes, Spacing, BorderRadius } from '../theme';

export default function StatusBadge({ status, label }) {
  const getStatusConfig = () => {
    switch (status) {
      case 'online':
      case 'success':
      case 'safe':
        return {
          color: Colors.success,
          icon: 'check-circle',
        };
      case 'offline':
      case 'error':
      case 'critical':
        return {
          color: Colors.danger,
          icon: 'close-circle',
        };
      case 'warning':
        return {
          color: Colors.warning,
          icon: 'alert-circle',
        };
      default:
        return {
          color: Colors.info,
          icon: 'information',
        };
    }
  };

  const config = getStatusConfig();

  return (
    <View style={[styles.container, { backgroundColor: `${config.color}20`, borderColor: `${config.color}40` }]}>
      <MaterialCommunityIcons 
        name={config.icon} 
        size={14} 
        color={config.color} 
      />
      <Text style={[styles.label, { color: config.color }]}>
        {label}
      </Text>
    </View>
  );
}

const styles = StyleSheet.create({
  container: {
    flexDirection: 'row',
    alignItems: 'center',
    paddingVertical: 4,
    paddingHorizontal: 8,
    borderRadius: BorderRadius.round,
    alignSelf: 'flex-start',
    borderWidth: 1,
  },
  label: {
    fontSize: FontSizes.xs,
    fontWeight: '700',
    marginLeft: 4,
    textTransform: 'uppercase',
    letterSpacing: 0.5,
  },
});
