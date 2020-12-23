import asyncio
import json
import websockets
import os
import sys
import numpy as np
import pandas as pd
import pprint
import ciso8601
from math import ceil
from math import floor
from time import time, sleep
from typing import Callable, Iterator
from passivbot_futures import load_key_secret, calc_new_ema, print_, ts_to_date, flatten, \
    filter_orders
import ccxt.async_support as ccxt_async


def date_to_ts(date: str):
    return ciso8601.parse_datetime(date).timestamp() * 1000


def round_up(n: float, step: float, safety_rounding=8) -> float:
    n_mod = n % step
    if n_mod == 0.0:
        return n
    return round(n - n_mod + step, safety_rounding)


def round_dn(n: float, step: float, safety_rounding=8) -> float:
    return round(n - n % step, safety_rounding)


async def create_bot(user: str, settings: str):
    bot = Bot(user, settings)
    await bot._init()
    return bot


class Bot:
    def __init__(self, user: str, settings: dict):
        self.settings = settings
        self.markup = settings['markup']
        self.symbol = settings['symbol']
        self.ema_span = settings['ema_span']
        self.spread = settings['spread']
        self.entry_amount = settings['entry_amount']
        self.leverage = settings['leverage']
        self.spread_minus = 1 - self.spread
        self.ema_bid_trigger_multiplier = 1 - self.spread * 0.99
        self.ema_ask_trigger_multiplier = 1 + self.spread * 0.99
        self.spread_plus = 1 + self.spread
        self.user = user
        self.ema = 0.0
        self.ema_alpha = 2 / (self.ema_span + 1)
        self.ema_alpha_ = 1 - self.ema_alpha
        self.price = 0.0
        self.cc = ccxt_async.bybit({'apiKey': (ks := load_key_secret('bybit', user))[0],
                                    'secret': ks[1]})
        self.ts_locked = {'decide': 0, 'update_state': 0, 'print': 0, 'create_bid': 0,
                          'create_ask': 0, 'cancel_orders': 0}
        self.ts_released = {k: self.ts_locked[k] + 1 for k in self.ts_locked}

    async def _init(self):
        info = await self.cc.public_get_symbols()
        for e in info['result']:
            if e['name'] == self.symbol:
                break
        else:
            raise Exception('symbol missing')
        self.price_step = float(e['price_filter']['tick_size'])
        self.amount_step = float(e['lot_size_filter']['qty_step'])

    async def update_state(self):
        position, _ = await asyncio.gather(
            self.cc.private_get_position_list(params={'symbol': self.symbol}),
            self.update_open_orders()
        )
        self.position = {'size': position['result']['size'],
                         'side': position['result']['side'],
                         'entry_price': float(position['result']['entry_price']),
                         'leverage': float(position['result']['leverage']),
                         'liq_price': float(position['result']['liq_price'])}
        self.ts_released['update_state'] = time()


    async def update_open_orders(self):
        try:
            open_orders = await self.cc.private_get_order(params={'symbol': self.symbol})
            self.open_orders = []
            self.highest_bid = 0.0
            self.lowest_ask = 9.9e9
            for o in open_orders['result']:
                self.open_orders.append({
                    'symbol': o['symbol'],
                    'side': o['side'],
                    'order_type': o['order_type'],
                    'price': float(o['price']),
                    'qty': float(o['qty']),
                    'order_id': o['order_id']
                })
                if o['side'] == 'Buy':
                    self.highest_bid = max(self.highest_bid, self.open_orders[-1]['price'])
                elif o['side'] == 'Sell':
                    self.lowest_ask = min(self.lowest_ask, self.open_orders[-1]['price'])
        except Exception as e:
            print(e)

    async def cancel_orders(self, orders: [dict]):
        if self.ts_locked['cancel_orders'] > self.ts_released['cancel_orders']:
            return
        self.ts_locked['cancel_orders'] = time()
        cancellations = []
        for o in orders:
            cancellations.append(self.cc.private_post_order_cancel(
                params={'symbol': o['symbol'], 'order_id': o['order_id']}
            ))
        try:
            cancelled = await asyncio.gather(*cancellations)
            for o in cancelled:
                try:
                    print_(['canceled order', o['result']['symbol'], o['result']['side'],
                            o['result']['qty'], o['result']['price']], n=True)
                except:
                    continue

        except Exception as e:
            print(e)
            cancelled = []
        await self.update_open_orders()
        self.ts_released['cancel_orders'] = time()
        return cancelled

    async def create_bid(self, amount: float, price: float):
        if self.ts_locked['create_bid'] > self.ts_released['create_bid']:
            return
        self.ts_locked['create_bid'] = time()
        try:
            o = await self.cc.private_post_order_create(
                params={'symbol': self.symbol, 'side': 'Buy', 'order_type': 'Limit',
                        'time_in_force': 'GoodTillCancel', 'qty': amount, 'price': price}
            )
            print_([' created order', self.symbol, o['result']['side'], o['result']['qty'],
                    o['result']['price']], n=True)
        except Exception as e:
            print('\n\nerror creating bid', amount, price, '\n\n')
            print(e)
            o = {}
        await self.update_state()
        self.ts_released['create_bid'] = time()
        return o

    async def create_ask(self, amount: float, price: float):
        if self.ts_locked['create_ask'] > self.ts_released['create_ask']:
            return
        self.ts_locked['create_ask'] = time()
        try:
            o = await self.cc.private_post_order_create(
                params={'symbol': self.symbol, 'side': 'Sell', 'order_type': 'Limit',
                        'time_in_force': 'GoodTillCancel', 'qty': amount, 'price': price}
            )
            print_([' created order', self.symbol, o['result']['side'], o['result']['qty'],
                    o['result']['price']], n=True)
        except Exception as e:
            print('\n\nerror creating ask', amount, price, '\n\n')
            print(e)
            o = {}
        await self.update_state()
        self.ts_released['create_ask'] = time()
        return o

    def calc_exit_double_down(self):
        if self.position['size'] == 0:
            return []
        elif self.position['side'] == 'Buy':
            # long position
            bid_price = round_up(max(
                self.position['entry_price'] * (1 - (1 / self.position['leverage']) / 2),
                self.position['liq_price'] + 0.00001
            ), self.price_step)
            ask_price = round_up(self.position['entry_price'] * (1 + self.markup), self.price_step)
        else:
            # shrt position
            ask_price = round_dn(min(
                self.position['entry_price'] * (1 + (1 / self.position['leverage']) / 2),
                self.position['liq_price'] - 0.00001
            ), self.price_step)
            bid_price = round_dn(self.position['entry_price'] * (1 - self.markup), self.price_step)
        return [{'symbol': self.symbol, 'side': 'Buy', 'qty': self.position['size'],
                 'price': bid_price},
                 {'symbol': self.symbol, 'side': 'Sell', 'qty': self.position['size'],
                 'price': ask_price}]

    async def init_ema(self) -> None:
        trades = await self.fetch_trades(self.symbol)
        additional_trades = await asyncio.gather(
            *[self.fetch_trades(self.symbol, from_id=trades[0]['trade_id'] - 1000 * i)
              for i in range(1, min(50, self.ema_span // 1000))])
        trades = sorted(trades + flatten(additional_trades), key=lambda x: x['trade_id'])
        ema = trades[0]['price']
        for t in trades:
            ema = ema * self.ema_alpha_ + t['price'] * self.ema_alpha
            r = t['price'] / ema
        self.price = t['price']
        self.ema = ema
        return trades

    async def fetch_trades(self, symbol: str, from_id: int = None) -> [dict]:
        params = {'symbol': symbol, 'limit': 1000}
        if from_id:
            params['from'] = from_id
        fetched_trades = await self.cc.public_get_trading_records(params=params)
        trades = [{'trade_id': t['id'],
                   'side': t['side'],
                   'price': t['price'],
                   'amount': t['qty'],
                   'timestamp': date_to_ts(t['time'][:-1])} for t in fetched_trades['result']]
        print_(['fetched trades', symbol, trades[0]['trade_id'],
                ts_to_date(trades[0]['timestamp'] / 1000)])
        return trades

    async def create_exits(self) -> list:
        to_cancel, to_create = filter_orders(self.open_orders,
                                             self.calc_exit_double_down(),
                                             keys=['side', 'qty', 'price'])
        tasks = []
        if to_cancel:
            tasks.append(self.cancel_orders(to_cancel))
        for o in to_create:
            if o['side'] == 'Buy':
                tasks.append(self.create_bid(o['qty'], o['price']))
            elif o['side'] == 'Sell':
                tasks.append(self.create_ask(o['qty'], o['price']))
        results = await asyncio.gather(*tasks)
        await asyncio.sleep(0.1)
        await self.update_state()
        if results:
            print()
        return results

    async def decide(self):
        if self.price <= self.highest_bid:
            self.ts_locked['decide'] = time()
            print_(['bid maybe taken'])
            await self.update_state()
            await self.create_exits()
            self.ts_released['decide'] = time()
            return
        elif self.price >= self.lowest_ask:
            self.ts_locked['decide'] = time()
            print_(['ask maybe taken'])
            await self.update_state()
            await self.create_exits()
            self.ts_released['decide'] = time()
            return
        elif self.position['size'] == 0:
            if self.price <= self.ema * self.ema_bid_trigger_multiplier:
                price = round_dn(min(self.ema, self.price) * self.spread_minus, self.price_step)
                if self.highest_bid != price:
                    self.ts_locked['decide'] = time()
                    await self.create_bid(self.entry_amount, price)
                    await asyncio.sleep(1.0)
                    await self.update_state()
                    await self.create_exits()
                    self.ts_released['decide'] = time()
                    return
            elif self.price >= self.ema * self.ema_ask_trigger_multiplier:
                price = round_up(max(self.ema, self.price) * self.spread_plus, self.price_step)
                if self.lowest_ask != price:
                    self.ts_locked['decide'] = time()
                    await self.create_ask(self.entry_amount, price)
                    await asyncio.sleep(1.0)
                    await self.update_state()
                    await self.create_exits()
                    self.ts_released['decide'] = time()
                    return
        if time() - self.ts_locked['decide'] > 5:
            self.ts_locked['decide'] = time()
            await self.update_state()
            await self.create_exits()
            self.ts_released['decide'] = time()
            return
        if time() - self.ts_released['print'] > 1:
            self.ts_released['print'] = time()
            line = f"{self.symbol} "
            if self.position['size'] == 0:
                bid_price = round_dn(self.ema * self.spread_minus, self.price_step)
                ask_price = round_up(self.ema * self.spread_plus, self.price_step)
                line += f"no position bid {bid_price} ask {ask_price} "
                r = 0.0
            elif self.position['side'] == 'Buy':
                line += f"long {self.position['size']} @ {self.position['entry_price']:.2f} "
                ddn_price, exit_price = 0.0, 0.0
                for o in self.open_orders:
                    if o['side'] == 'Buy':
                        ddn_price = o['price']
                    elif o['side'] == 'Sell':
                        exit_price = o['price']
                line += f"exit {exit_price} ddown {ddn_price} "
                try:
                    r = (self.price - ddn_price) / (exit_price - ddn_price)
                except ZeroDivisionError:
                    r = 0.5
            elif self.position['side'] == 'Sell':
                line += f"shrt {self.position['size']} @ {self.position['entry_price']:.2f} "
                ddn_price, exit_price = 0.0, 0.0
                for o in self.open_orders:
                    if o['side'] == 'Sell':
                        ddn_price = o['price']
                    elif o['side'] == 'Buy':
                        exit_price = o['price']
                try:
                    r = 1 - (self.price - exit_price) / (ddn_price - exit_price)
                except ZeroDivisionError:
                    r = 0.5
                line += f"exit {exit_price} ddown {ddn_price} "

            line += f"pct {r:.2f} last {self.price}   "
            print_([line], r=True)

    async def start_websocket(self) -> None:
        self.stop_websocket = False
        uri = f"wss://stream.bybit.com/realtime"
        print_([uri])
        await self.update_state()
        if self.position['leverage'] != self.leverage:
            print(await self.cc.user_post_leverage_save(
                params={'symbol': self.symbol, 'leverage': 0}
            ))
        await self.init_ema()
        param = {'op': 'subscribe', 'args': ['trade.' + self.symbol]}
        k = 1
        async with websockets.connect(uri) as ws:
            print('starting websocket')
            await ws.send(json.dumps(param))
            print('started websocket')

            async for msg in ws:
                if msg is None:
                    continue
                data = json.loads(msg)
                try:
                    for e in data['data']:
                        self.ema = calc_new_ema(self.price,
                                                e['price'],
                                                self.ema,
                                                alpha=self.ema_alpha)
                        self.price = e['price']
                except Exception as e:
                    if 'success' not in data:
                        print(e)
                if self.ts_locked['decide'] < self.ts_released['decide']:
                    asyncio.create_task(self.decide())
                elif k % 10 == 0:
                    self.flush_stuck_locks()
                    k = 1
                k += 1

    def flush_stuck_locks(self, timeout: float = 4.0) -> None:
        now = time()
        for key in self.ts_locked:
            if self.ts_locked[key] > self.ts_released[key]:
                if now - self.ts_locked[key] > timeout:
                    print('flushing', key)
                    self.ts_released[key] = now




async def main():
    await start_bot()


async def start_bot(n_tries: int = 0):
    user = sys.argv[1]
    settings = {'symbol': 'BTCUSD', 'markup': 0.00143, 'leverage': 100, 'ema_span': 38036,
                'spread': 0.00011, 'entry_amount': 1}
    max_n_tries = 30
    try:
        bot = await create_bot(user, settings)
        await bot.start_websocket()
    except KeyboardInterrupt:
        await bot.cc.close()
    except Exception as e:
        await bot.cc.close()
        print(e)
        if n_tries >= max_n_tries:
            return
        n_tries += 1
        for k in range(10, -1, -1):
            sys.stdout.write(f'\rrestarting bot in {k} seconds   ')
            sleep(1)
        await start_bot(n_tries + 1)


if __name__ == '__main__':
    asyncio.run(main())

