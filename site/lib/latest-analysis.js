export class LatestAnalysis {
  constructor({run, publish, fail, pending, delay = 450}) {
    Object.assign(this, {run, publish, fail, pending, delay});
    this.revision = 0;
    this.active = false;
    this.enabled = false;
    this.draft = '';
  }
  update(draft) {
    this.draft = draft;
    this.revision++;
    clearTimeout(this.timer);
    this.pending(!!draft.trim());
    if (this.enabled && draft.trim()) this.timer = setTimeout(() => this.pump(), this.delay);
  }
  setEnabled(enabled) {
    this.enabled = enabled;
    this.update(this.draft);
  }
  async pump() {
    if (this.active || !this.enabled || !this.draft.trim()) return;
    const revision = this.revision, draft = this.draft;
    this.active = true;
    try {
      const result = await this.run(draft);
      if (this.enabled && revision === this.revision) this.publish(result, draft);
    } catch (error) {
      if (this.enabled && revision === this.revision) this.fail(error);
    } finally {
      this.active = false;
      if (this.enabled && revision !== this.revision && this.draft.trim()) {
        clearTimeout(this.timer);
        this.timer = setTimeout(() => this.pump(), this.delay);
      }
    }
  }
}
