class CreateEstimations < ActiveRecord::Migration[8.0]
  def change
    create_table :estimations do |t|
      t.text     :transcription,    null: false
      t.integer  :hourly_rate,      null: false, default: 40
      t.jsonb    :response_payload, null: false, default: {}
      t.string   :prompt_version
      t.boolean  :cached,           null: false, default: false

      t.timestamps
    end

    add_index :estimations, :created_at
  end
end
