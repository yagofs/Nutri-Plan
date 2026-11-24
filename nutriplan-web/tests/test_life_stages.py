"""Tests for heifer and dry-cow life stages."""
import unittest

from services.milk_targets import formulate
from services.native_ration import optimize_native
from services.nutrition import calculate, categories


def heifer_payload(category='heifer_I', **parameter_overrides):
    parameters = {
        'mature_body_weight': 700,
        'body_condition_score_5': 3,
        'average_daily_gain_heifer': 0.8,
        'housing': 'Barn',
        'distance': 0,
        'ndf_percentage': 30,
    }
    parameters.update(parameter_overrides)
    return {'animal_category': category, 'body_weight_kg': 300, 'parameters': parameters}


def dry_cow_payload(**parameter_overrides):
    parameters = {
        'mature_body_weight': 740,
        'body_condition_score_5': 3,
        'calving_interval': 400,
        'day_of_pregnancy': 220,
        'housing': 'Barn',
        'distance': 0,
        'ndf_percentage': 30,
    }
    parameters.update(parameter_overrides)
    return {'animal_category': 'dry_cow', 'body_weight_kg': 700, 'parameters': parameters}


class LifeStageRequirementTests(unittest.TestCase):
    def test_categories_include_life_stages(self):
        identifiers = {item['id'] for item in categories()}
        self.assertTrue({'heifer_I', 'heifer_II', 'heifer_III', 'dry_cow'} <= identifiers)

    def test_heifers_need_average_daily_gain(self):
        payload = heifer_payload()
        payload['parameters'].pop('average_daily_gain_heifer')
        result = calculate(payload)
        self.assertEqual(result['status'], 'incomplete_input')
        self.assertIn('average_daily_gain_heifer', result['missing_inputs'])

    def test_heifer_stages_succeed(self):
        for category in ('heifer_I', 'heifer_II', 'heifer_III'):
            result = calculate(heifer_payload(category))
            self.assertEqual(result['status'], 'success', result)
            self.assertGreater(result['requirements']['net_energy'], 0)
            self.assertIsNone(result['parameters']['days_in_milk'])
            self.assertEqual(result['parameters']['milk_production'], 0)

    def test_heifer_one_cannot_be_pregnant(self):
        result = calculate(heifer_payload(day_of_pregnancy=40))
        self.assertEqual(result['status'], 'invalid_input')
        self.assertEqual(result['error_code'], 'pregnancy_not_applicable')

    def test_dry_cow_needs_calving_interval(self):
        payload = dry_cow_payload()
        payload['parameters'].pop('calving_interval')
        result = calculate(payload)
        self.assertEqual(result['status'], 'incomplete_input')
        self.assertIn('calving_interval', result['missing_inputs'])

    def test_dry_cow_succeeds(self):
        result = calculate(dry_cow_payload())
        self.assertEqual(result['status'], 'success', result)
        self.assertGreater(result['requirements']['dry_matter'], 0)


class LifeStageOptimizationTests(unittest.TestCase):
    def test_heifer_and_dry_cow_formulate_without_milk_floor(self):
        cases = [
            (heifer_payload('heifer_I'), 'growth'),
            (heifer_payload('heifer_III'), 'growth'),
            (dry_cow_payload(), 'dry'),
        ]
        for payload, stage in cases:
            result = optimize_native(payload)
            self.assertTrue(result.get('ok'), result.get('message'))
            self.assertEqual(result['stage_type'], stage)
            self.assertEqual(len(result['constraints']), 11)
            self.assertTrue(all(item['satisfied'] for item in result['constraints']))
            self.assertIsNone(result.get('milk_potential'))
            self.assertNotIn('target_results', result)

    def test_non_lactating_derives_ndf_from_optimized_ration(self):
        payload = heifer_payload('heifer_I')
        payload['parameters'].pop('ndf_percentage')
        result = optimize_native(payload)
        self.assertTrue(result.get('ok'), result.get('message'))
        self.assertAlmostEqual(
            result['ration_context']['ndf_percentage'],
            100 * result['nutrient_supply']['ndf']['value'] / result['dmi'],
            places=3,
        )

    def test_milk_target_scan_is_bypassed_for_non_lactating(self):
        payload = heifer_payload('heifer_II')
        payload['evaluate_milk_targets'] = True
        result = formulate(payload)
        self.assertTrue(result.get('ok'), result.get('message'))
        self.assertNotIn('target_results', result)
        self.assertNotIn('frontier', result)


if __name__ == '__main__':
    unittest.main()
