use super::*;

#[cfg(feature = "decima")]
impl Backend for crate::decima::Model {
    type Parameters = DecimaParameters;
    type Cache = crate::decima::Session;
    fn capabilities(&self) -> Capabilities {
        Capabilities {
            model: "decima",
            backend: "cpu",
            caches: vec!["choice"],
            tuning: "heuristic",
        }
    }
    fn create_cache(&self, options: &CacheOptions) -> Result<Self::Cache> {
        let mut cache = Self::Cache::default();
        cache.set_cache_bytes(options.max_bytes);
        Ok(cache)
    }
    fn clear_cache(&self, cache: &mut Self::Cache) {
        cache.clear_cache();
    }
    async fn predict(
        &mut self,
        request: &Request,
        parameters: &Self::Parameters,
        cache: &mut Self::Cache,
        _: &Cancellation,
    ) -> Result<Value> {
        self.predict_json(&parameters.request(request), cache)
            .map_err(Error::engine)
    }
}

#[cfg(feature = "laya")]
impl Backend for crate::laya::Model {
    type Parameters = LayaParameters;
    type Cache = crate::laya::Scratch;
    fn capabilities(&self) -> Capabilities {
        Capabilities {
            model: "laya",
            backend: "cpu",
            caches: vec![],
            tuning: "heuristic",
        }
    }
    fn create_cache(&self, _: &CacheOptions) -> Result<Self::Cache> {
        Ok(Self::Cache::default())
    }
    fn clear_cache(&self, cache: &mut Self::Cache) {
        cache.clear_token_cache();
        cache.clear_state_cache();
    }
    async fn predict(
        &mut self,
        request: &Request,
        parameters: &Self::Parameters,
        cache: &mut Self::Cache,
        _: &Cancellation,
    ) -> Result<Value> {
        self.predict_json(parameters.request(request), cache)
            .map_err(Error::engine)
    }
}

#[cfg(all(feature = "decima", feature = "webgpu"))]
impl Backend for crate::decima::webgpu::Model {
    type Parameters = DecimaParameters;
    type Cache = crate::decima::webgpu::Session;
    fn capabilities(&self) -> Capabilities {
        Capabilities {
            model: "decima",
            backend: "webgpu",
            caches: vec!["choice"],
            tuning: "heuristic",
        }
    }
    fn create_cache(&self, options: &CacheOptions) -> Result<Self::Cache> {
        let mut cache = Self::Cache::default();
        cache.set_cache_bytes(options.max_bytes);
        Ok(cache)
    }
    fn clear_cache(&self, cache: &mut Self::Cache) {
        cache.clear_cache();
    }
    async fn predict(
        &mut self,
        request: &Request,
        parameters: &Self::Parameters,
        cache: &mut Self::Cache,
        _: &Cancellation,
    ) -> Result<Value> {
        self.predict_json(&parameters.request(request), cache)
            .await
            .map_err(Error::engine)
    }
}

#[cfg(all(feature = "laya", feature = "webgpu"))]
impl Backend for crate::laya::webgpu::Model {
    type Parameters = LayaParameters;
    type Cache = ();
    fn capabilities(&self) -> Capabilities {
        Capabilities {
            model: "laya",
            backend: "webgpu",
            caches: vec![],
            tuning: "heuristic",
        }
    }
    fn create_cache(&self, _: &CacheOptions) -> Result<()> {
        Ok(())
    }
    fn clear_cache(&self, _: &mut ()) {}
    async fn predict(
        &mut self,
        request: &Request,
        parameters: &Self::Parameters,
        _: &mut (),
        _: &Cancellation,
    ) -> Result<Value> {
        self.predict_json(parameters.request(request))
            .await
            .map_err(Error::engine)
    }
}

macro_rules! prefix_backend {
    ($module:ident, $parameters:ty, $name:literal) => {
        impl Backend for crate::$module::Model {
            type Parameters = $parameters;
            type Cache = crate::$module::Session;
            fn capabilities(&self) -> Capabilities {
                Capabilities {
                    model: $name,
                    backend: "cpu",
                    caches: vec!["prefix"],
                    tuning: "heuristic",
                }
            }
            fn create_cache(&self, options: &CacheOptions) -> Result<Self::Cache> {
                let mut cache = Self::Cache::default();
                cache.set_cache_bytes(options.max_bytes);
                Ok(cache)
            }
            fn clear_cache(&self, cache: &mut Self::Cache) {
                cache.clear_cache();
            }
            async fn predict(
                &mut self,
                request: &Request,
                _: &Self::Parameters,
                cache: &mut Self::Cache,
                _: &Cancellation,
            ) -> Result<Value> {
                self.predict_json(&request.value(), cache)
                    .map_err(Error::engine)
            }
        }
    };
}
#[cfg(feature = "kev")]
prefix_backend!(kev, KevParameters, "kev");
#[cfg(feature = "jev-style")]
prefix_backend!(jev_style, JevParameters, "jev-style");

#[cfg(all(feature = "pplx-flash", feature = "cpu"))]
impl Backend for crate::pplx_flash::Model {
    type Parameters = PplxParameters;
    type Cache = ();
    fn capabilities(&self) -> Capabilities {
        Capabilities {
            model: "pplx-flash",
            backend: "cpu",
            caches: vec![],
            tuning: "heuristic",
        }
    }
    fn create_cache(&self, _: &CacheOptions) -> Result<()> {
        Ok(())
    }
    fn clear_cache(&self, _: &mut ()) {}
    async fn predict(
        &mut self,
        request: &Request,
        parameters: &PplxParameters,
        _: &mut (),
        _: &Cancellation,
    ) -> Result<Value> {
        parameters.validate()?;
        crate::pplx_flash::Model::predict(self, &request.value(), parameters.max_sequence_tokens)
            .map_err(Error::engine)
    }
}
