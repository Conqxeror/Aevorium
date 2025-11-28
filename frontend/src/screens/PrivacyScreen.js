import React, { useEffect, useState } from 'react';
import { View, Text, StyleSheet, Alert } from 'react-native';
import { Button, Card, TextInput } from 'react-native-paper';
import api from '../components/ApiClient';

export default function PrivacyScreen() {
  const [summary, setSummary] = useState(null);
  const [newBudget, setNewBudget] = useState('100');
  const [loading, setLoading] = useState(false);

  useEffect(()=>{
    fetchSummary();
  },[])

  const fetchSummary = async () => {
    try {
      const r = await api.get('/privacy-budget');
      setSummary(r.data);
    } catch (e) {
      setSummary(null);
    }
  }

  const setBudget = async () => {
    try {
      const budget = Number(newBudget);
      if (isNaN(budget) || budget < 0) {
        Alert.alert('Invalid budget', 'Enter a valid positive epsilon');
        return;
      }
      setLoading(true);
      const r = await api.post('/privacy-budget/set-limit', { total_budget: budget });
      if (r.status === 200) {
        Alert.alert('Success', 'Privacy budget set');
        fetchSummary();
      } else {
        Alert.alert('Failed', `Status ${r.status}`);
      }
    } catch (e) {
      Alert.alert('Error', e.message || 'Failed to set budget');
    } finally {
      setLoading(false);
    }
  }

  const resetBudget = async () => {
    try {
      setLoading(true);
      const r = await api.post('/privacy-budget/reset');
      if (r.status === 200) {
        Alert.alert('Success', 'Privacy budget reset');
        fetchSummary();
      } else {
        Alert.alert('Failed', `Status ${r.status}`);
      }
    } catch (e) {
      Alert.alert('Error', e.message || 'Failed to reset');
    } finally {
      setLoading(false);
    }
  }

  return (
    <View style={styles.container}>
      <Card style={styles.card}>
        <Card.Title title="Privacy Budget" />
        <Card.Content>
          {summary ? (
            <View>
              <Text>Cumulative ε: {summary.cumulative_epsilon.toFixed(2)}</Text>
              <Text>Delta: {String(summary.delta)}</Text>
              <Text>Total budget: {summary.total_budget === null ? '∞' : summary.total_budget}</Text>
            </View>
          ) : (
            <Text>No privacy summary available</Text>
          )}
        </Card.Content>
      </Card>

      <Card style={styles.card}>
        <Card.Title title="Set Budget" />
        <Card.Content>
          <TextInput label="Total ε" value={newBudget} onChangeText={setNewBudget} keyboardType='numeric' />
          <Button mode='contained' onPress={setBudget} style={{marginTop:10}} loading={loading}>
            Set Budget
          </Button>
        </Card.Content>
      </Card>

      <Card style={styles.card}>
        <Card.Title title="Reset Budget" />
        <Card.Content>
          <Button mode='outlined' onPress={resetBudget} loading={loading}>
            Reset
          </Button>
        </Card.Content>
      </Card>
    </View>
  )
}

const styles = StyleSheet.create({ container: { padding: 10, flex: 1 }, card: { marginBottom: 8 } });
