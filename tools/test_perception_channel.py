import unittest
from unittest.mock import patch
from perception_trial import *

class SensorChannelContract(unittest.TestCase):
    def setUp(self):self.base=conditions()[0];self.scene=make_scene('empty')
    def channel(self,**kw):return Channel(dict(self.base,**kw),42)
    def test_delay_preserves_capture_pose_and_timestamp(self):
        c=self.channel(delay_s=.2);c.capture(self.scene,[-7,0,2.5],1.)
        self.assertIsNone(c.deliver(1.19));x=c.deliver(1.2)
        self.assertEqual(x['stamp'],1.);self.assertEqual(x['origin'],[-7,0,2.5]);c.used(x,1.2)
    def test_no_future_scan_access(self):
        c=self.channel(delay_s=.2);c.capture(self.scene,[-7,0,2.5],1.)
        with self.assertRaises(AssertionError):c.used(c.queue[0][2],1.1)
    def test_full_drop_delivers_nothing(self):
        c=self.channel(drop_probability=1.);c.capture(self.scene,[-7,0,2.5],1.)
        self.assertIsNone(c.deliver(2.));self.assertTrue(c.records[0]['dropped'])
    def test_latest_received_replaces_unconsumed(self):
        c=self.channel();c.capture(self.scene,[-7,0,2.5],1.);c.capture(self.scene,[-6.9,0,2.5],1.2)
        self.assertEqual(c.deliver(1.2)['sequence'],1);self.assertIsNone(c.records[0]['mapped_at'])
    def test_scan_cannot_be_consumed_twice(self):
        c=self.channel();c.capture(self.scene,[-7,0,2.5],1.);x=c.deliver(1.);c.used(x,1.)
        with self.assertRaises(AssertionError):c.used(x,1.)
    def test_fixed_point_budget_is_constant(self):
        cc=[x for x in conditions() if x['family']=='rate_fixed_points_per_second']
        self.assertEqual({x['rate_hz']*x['azimuths']*17 for x in cc},{61200.})
    def test_point_density_preserves_fov(self):
        for az in [90,180,360,720,1440]:
            d=Sensor(az).d
            self.assertEqual(len(d),az*17);self.assertAlmostEqual(d[:,2].min(),np.sin(np.deg2rad(-8)))
            self.assertAlmostEqual(d[:,2].max(),np.sin(np.deg2rad(8)))
    def test_seeded_loss_is_nested(self):
        a=self.channel(drop_probability=.1);b=self.channel(drop_probability=.5)
        for i in range(10):
            a.capture(self.scene,[-7,0,2.5],1.+i*.2);b.capture(self.scene,[-7,0,2.5],1.+i*.2)
        self.assertEqual([x['uniform'] for x in a.records],[x['uniform'] for x in b.records])
        self.assertTrue(all(not x['dropped'] or y['dropped'] for x,y in zip(a.records,b.records)))
    def test_burst_loss_interval(self):
        c=self.channel(burst_period_s=2.,burst_length_s=.6)
        for i in range(11):c.capture(self.scene,[-7,0,2.5],1.+i*.2)
        self.assertEqual([x['sequence'] for x in c.records if x['dropped']],[0,1,2,10])
    def test_capture_schedule_has_no_frequency_drift(self):
        for rate in [2.5,5.,10.,20.]:
            c=self.channel(rate_hz=rate)
            for tick in range(100):
                t=round(1.+tick*.01,10)
                if c.due(t):c.capture(self.scene,[-7,0,2.5],t)
            self.assertEqual(len(c.records),int(np.ceil(rate)))

if __name__=='__main__':unittest.main(verbosity=2)
