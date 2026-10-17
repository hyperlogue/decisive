use anyhow::{Result, ensure};
use serde::{Deserialize, Serialize};

#[derive(Clone, Debug, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct Band<T> {
    // Inclusive actual-token bound, not a padding target.
    pub through: usize,
    pub settings: T,
}

#[derive(Clone, Debug, Serialize)]
pub struct Rules<T> {
    bands: Vec<Band<T>>,
    fallback: T,
}
impl<'de, T: Deserialize<'de>> Deserialize<'de> for Rules<T> {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        #[derive(Deserialize)]
        #[serde(deny_unknown_fields)]
        struct Data<T> {
            bands: Vec<Band<T>>,
            fallback: T,
        }
        let data = Data::deserialize(deserializer)?;
        Self::new(data.bands, data.fallback).map_err(serde::de::Error::custom)
    }
}
impl<T> Rules<T> {
    pub fn new(bands: Vec<Band<T>>, fallback: T) -> Result<Self> {
        let rules = Self { bands, fallback };
        rules.validate(|_| Ok(()))?;
        Ok(rules)
    }
    pub fn validate(&self, check: impl Fn(&T) -> Result<()>) -> Result<()> {
        let mut previous = 0;
        for band in &self.bands {
            ensure!(
                band.through > previous,
                "length bounds must be positive and strictly increasing"
            );
            check(&band.settings)?;
            previous = band.through;
        }
        check(&self.fallback)
    }
    pub fn select(&self, tokens: usize) -> &T {
        self.bands
            .get(self.bands.partition_point(|b| b.through < tokens))
            .map_or(&self.fallback, |b| &b.settings)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn inclusive_boundaries_and_unbounded_fallback() -> Result<()> {
        let rules = Rules::new(
            vec![
                Band {
                    through: 64,
                    settings: 1,
                },
                Band {
                    through: 512,
                    settings: 2,
                },
            ],
            3,
        )?;
        for (length, expected) in [
            (0, 1),
            (1, 1),
            (64, 1),
            (65, 2),
            (512, 2),
            (513, 3),
            (usize::MAX, 3),
        ] {
            assert_eq!(*rules.select(length), expected);
        }
        assert!(
            Rules::new(
                vec![Band {
                    through: 0,
                    settings: 1
                }],
                2
            )
            .is_err()
        );
        assert!(
            Rules::new(
                vec![
                    Band {
                        through: 2,
                        settings: 1
                    },
                    Band {
                        through: 2,
                        settings: 2
                    }
                ],
                3
            )
            .is_err()
        );
        Ok(())
    }
}
