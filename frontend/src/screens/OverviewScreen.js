import React, { useEffect, useState } from 'react';
import { View, Text, StyleSheet, ScrollView } from 'react-native';
import { Button, Headline, Subheading, Card, TextInput } from 'react-native-paper';
import api, { setApiToken } from '../components/ApiClient';

export default function OverviewScreen() {
  const [apiOnline, setApiOnline] = useState(false);
  const [privacy, setPrivacy] = useState(null);
  const [logs, setLogs] = useState([]);
  const [apiToken, setApiTokenField] = useState('');

  useEffect(() => {
    checkApiHealth();
    loadPrivacy();
    loadLogs();
  }, []);

  useEffect(() => {
    setApiToken(apiToken);
  }, [apiToken]);

  const checkApiHealth = async () => {
    try {
      const r = await api.get('/health');
      if (r.status === 200) setApiOnline(true);
    } catch (e) {
      setApiOnline(false);
    }
  };

  const loadPrivacy = async () => {
    try {
      const r = await api.get('/privacy-budget');
      setPrivacy(r.data);
    } catch (e) {
      setPrivacy(null);
    }
  };

  const loadLogs = async () => {
    try {
      const r = await api.get('/audit-log');
      setLogs(r.data || []);
    } catch (e) {
      setLogs([]);
    }
  };

  return (
    <ScrollView style={styles.container}>
      <Headline style={{ marginTop: 10 }}>Aevorium Overview</Headline>
      <Card style={styles.card}>
        <Card.Title title="API Status" />
        <Card.Content>
          <Text>{apiOnline ? 'Online' : 'Offline'}</Text>
        </Card.Content>
      </Card>

      <Card style={styles.card}>
        <Card.Title title="Settings" />
        <Card.Content>
          <Text>Set API Token (optional)</Text>
          <TextInput placeholder="Bearer token" value={apiToken} onChangeText={setApiTokenField} style={{marginTop:8}} />
        </Card.Content>
      </Card>

      <Card style={styles.card}>
        <Card.Title title="Privacy Budget" />
        <Card.Content>
          {privacy ? (
            <View>
              <Subheading> Cumulative ε: {privacy.cumulative_epsilon.toFixed(2)}</Subheading>
              <Subheading> Remaining: {privacy.budget_remaining === Infinity ? '∞' : privacy.budget_remaining.toFixed(2)}</Subheading>
            </View>
          ) : (
            <Text>No privacy data</Text>
          )}
        </Card.Content>
      </Card>

      <Card style={styles.card}>
        <Card.Title title="Recent Activity" />
        <Card.Content>
          {logs && logs.length > 0 ? (
            logs.slice(-5).reverse().map((l,i)=> (
              <Text key={i}>{l.timestamp} - {l.event_type}</Text>
            ))
          ) : (
            <Text>No events yet</Text>
          )}
        </Card.Content>
      </Card>

    </ScrollView>
  );
}

const styles = StyleSheet.create({
  container: { flex: 1, padding: 10 },
  card: { marginVertical: 8 }
});
